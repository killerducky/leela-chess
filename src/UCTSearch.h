/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <memory>
#include <atomic>
#include <tuple>
#include <unordered_map>

#include "Position.h"
#include "UCTNode.h"

// SearchResult is in [0,1]
// 0.0 represents Black win
// 0.5 represents draw
// 1.0 represents White win
// Eg. 0.1 would be a high probability of Black winning.
class SearchResult {
public:
    SearchResult() = default;
    bool valid() const { return m_valid;  }
    float eval() const { return m_eval;  }
    static SearchResult from_eval(float eval) {
        return SearchResult(eval);
    }
    static SearchResult from_score(float board_score) {
        if (board_score > 0.0f) {
            return SearchResult(1.0f);
        } else if (board_score < 0.0f) {
            return SearchResult(0.0f);
        } else {
            return SearchResult(0.5f);
        }
    }
private:
    explicit SearchResult(float eval)
        : m_valid(true), m_eval(eval) {}
    bool m_valid{false};
    float m_eval{0.0f};
};

class UCTSearch {
public:
    /*
        Maximum size of the tree in memory. Nodes are about
        40 bytes, so limit to ~1.6G.
    */
    static constexpr auto MAX_TREE_SIZE = 40'000'000;

    UCTSearch(BoardHistory&& bh);
    Move think(BoardHistory&& bh);
    void set_playout_limit(int playouts);
    void set_analyzing(bool flag);
    void set_quiet(bool flag);
    void ponder();
    bool is_running() const;
    bool playout_limit_reached() const;
    void increment_playouts();
    SearchResult play_simulation(BoardHistory& bh, UCTNode* const node);
    
private:
    void dump_stats(BoardHistory& pos, UCTNode& parent);
    std::string get_pv(BoardHistory& pos, UCTNode& parent);
    void dump_analysis(int elapsed, bool force_output);
    Move get_best_move();

    BoardHistory bh_;
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_nodes{0};
    std::atomic<int> m_playouts{0};
    std::atomic<bool> m_run{false};
    int m_maxplayouts;

    bool quiet_ = true;
};

class UCTWorker {
public:
    UCTWorker(const BoardHistory& bh, UCTSearch* search, UCTNode* root)
      : bh_(bh), m_search(search), m_root(root) {}
    void operator()();
private:
    const BoardHistory& bh_;
    UCTSearch* m_search;
    UCTNode* m_root;
};

#endif
