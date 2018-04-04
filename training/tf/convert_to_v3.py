#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

import binascii
import glob
import gzip
import itertools
import math
import multiprocessing as mp
import numpy as np
import random
import shufflebuffer as sb
import struct
import sys
import threading
import time
import tensorflow as tf
import unittest

# VERSION of the training data file format
#     1 - Text, oldflip
#     2 - Binary, oldflip
#     3 - Binary, newflip
#     b'\1\0\0\0' - Invalid, see issue #119
#
# Note: VERSION1 does not include a version in the header, it starts with
# text hex characters. This means any VERSION that is also a valid ASCII
# hex string could potentially be a training file. Most of the time it
# will be "00ff", but maybe games could get split and then there are more
# "00??" possibilities.
#
# Also note "0002" is actually b'\0x30\0x30\0x30\0x32' (or maybe reversed?)
# so it doesn't collide with VERSION2.
#
VERSION1 = struct.pack('i', 1)
VERSION2 = struct.pack('i', 2)

NUM_HIST = 8
NUM_PLANES = 14
NUM_REALS = 7  # 4 castling, 1 color, 1 50rule, 1 movecount
NUM_OUTPUTS = 2 # policy, value
# 14*8 planes,
DATA_ITEM_LINES = NUM_HIST*NUM_PLANES+NUM_REALS+NUM_OUTPUTS # 121
# Note: The C++ code adds 1 unused REAL to make the number even.

class Board:
    def __init__(self):
        self.clear_board()

    def clear_board(self):
        self.board = []
        for rank in range(8):
            self.board.append(list("."*8))
        self.reps = 0

    def describe(self):
        s = []
        for rank in range(8):
            s.append("".join(self.board[rank]))
        s.append("reps {}  ".format(self.reps))
        return s

class TrainingStep:
    def __init__(self):
        # Build a series of flat planes with values 0..255
        self.flat_planes = []
        for i in range(256):
            self.flat_planes.append(bytes([i]*64))
        self.init_structs()
        self.history = []
        for history in range(NUM_HIST):
            self.history.append(Board())
        self.us_ooo = 0
        self.us_oo  = 0
        self.them_ooo = 0
        self.them_oo  = 0
        self.us_black = 0
        self.rule50_count = 0

    def init_structs(self):
        # struct.Struct doesn't pickle, so it needs to be separately
        # constructed in workers.

        # V2 Format (8604 bytes total)
        # int32 version (4 bytes)
        # 1924 float32 probabilities (7696 bytes)
        # 112 packed bit planes (896 bytes)
        # uint8 castling us_ooo (1 byte)
        # uint8 castling us_oo (1 byte)
        # uint8 castling them_ooo (1 byte)
        # uint8 castling them_oo (1 byte)
        # uint8 side_to_move (1 byte)
        # uint8 rule50_count (1 byte)
        # uint8 move_count (1 byte)
        # int8 result (1 byte)
        self.v2_struct = struct.Struct('4s7696s896sBBBBBBBb')

    def clear_hist(self):
        for hist in range(NUM_HIST):
            self.history.clear_board()

    def update_board(self, hist, piece, hex_string):
        """
            Update the ASCII board representation
        """
        bit_board = int(hex_string, 16)
        for r in range(8):
            for f in range(8):
                if bit_board & (1<<(r*8+f)):
                    assert(self.history[hist].board[r][f] == ".")
                    self.history[hist].board[r][f] = piece
                    #self.history[hist].board[r] = \
                    #    self.history[hist].board[r][:f] + \
                    #    piece + \
                    #    self.history[hist].board[r][8-f:]
                    
    def describe(self):
        s = ""
        if self.us_black:
            s += "us = Black\n"
        else:
            s += "us = White\n"
        s += "rule50_count {} b_ooo b_oo, w_ooo, w_oo {} {} {} {}\n".format(
            self.rule50_count, self.us_ooo, self.us_oo, self.them_ooo, self.them_oo)
        rank_strings = []
        for hist in range(NUM_HIST):
            rank_strings.append(self.history[hist].describe())
        for rank in range(8):
            for hist in range(NUM_HIST):
                s += rank_strings[hist][rank] + " "
            s += "\n"
        return s

    def update_reals(self, text_item):
        self.us_ooo = int(text_item[NUM_HIST*NUM_PLANES+0])
        self.us_oo = int(text_item[NUM_HIST*NUM_PLANES+1])
        self.them_ooo =int(text_item[NUM_HIST*NUM_PLANES+2])
        self.them_oo = int(text_item[NUM_HIST*NUM_PLANES+3])
        self.us_black =int(text_item[NUM_HIST*NUM_PLANES+4])
        self.rule50_count = min(int(text_item[NUM_HIST*NUM_PLANES+5]), 255)  # TODO should be around 99-102ish

    def convert_v1_to_v3(self, ply, text_item):
        """
            Convert v1 text format to v2 packed binary format

            Converts a chess chunk of ascii text into a byte string
            [[plane_1],[plane_2],...],...
            [probabilities],...
            winner,...
        """
        # We start by building a list of 112 planes, each being a 8*8=64
        # element array of type np.uint8
        planes = []
        for hist in range(NUM_HIST):
            # Us   -- uppercase
            # Them -- lowercase
            self.update_board(hist, "P", text_item[hist*NUM_PLANES+0])
            self.update_board(hist, "N", text_item[hist*NUM_PLANES+1])
            self.update_board(hist, "B", text_item[hist*NUM_PLANES+2])
            self.update_board(hist, "R", text_item[hist*NUM_PLANES+3])
            self.update_board(hist, "Q", text_item[hist*NUM_PLANES+4])
            self.update_board(hist, "K", text_item[hist*NUM_PLANES+5])
            self.update_board(hist, "p", text_item[hist*NUM_PLANES+6])
            self.update_board(hist, "n", text_item[hist*NUM_PLANES+7])
            self.update_board(hist, "b", text_item[hist*NUM_PLANES+8])
            self.update_board(hist, "r", text_item[hist*NUM_PLANES+9])
            self.update_board(hist, "q", text_item[hist*NUM_PLANES+10])
            self.update_board(hist, "k", text_item[hist*NUM_PLANES+11])
            if (text_item[hist*NUM_PLANES+12] != "0000000000000000"):
                self.history[hist].reps = 1
                assert (text_item[hist*NUM_PLANES+12] == "ffffffffffffffff")
            # It's impossible to have a position in training with reps=2
            # Because that is already a draw.
            assert (text_item[hist*NUM_PLANES+13] == "0000000000000000")

        # Now we extract the non plane information
        self.update_reals(text_item)
        print("ply {} move {} (Not actually part of training data)".format(
            ply+1, (ply+2)//2))
        print(self.describe())

def test_hist_flip(files):
    """
        Test flipping history
    """
    for filename in files:
        print("Parsing {}".format(filename))
        with gzip.open(filename, 'rt') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
            for i in range(0, len(lines), DATA_ITEM_LINES):
                text_item = lines[i:i+DATA_ITEM_LINES]
                ts = TrainingStep()
                ts.convert_v1_to_v3(i//DATA_ITEM_LINES, text_item)

if __name__ == '__main__':
    test_hist_flip(sys.argv[1:])
