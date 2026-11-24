import numpy as np

from paper_soccer.agents.tabular_q import TabularQAgent
from paper_soccer.evaluation.arena import evaluate_vs, play_game
from paper_soccer.game import PaperSoccerGame
from paper_soccer.opponents import GreedyOpponent, RandomOpponent


def test_opponents_only_choose_legal_actions():
    rng = np.random.default_rng(0)
    for opp in (RandomOpponent(), GreedyOpponent()):
        g = PaperSoccerGame(6, 8)
        for _ in range(10):
            g.reset()
            end = False
            while not end:
                a = opp.act(g, rng)
                assert a in g.legal_actions()
                _, _, end, _ = g.step(a, copy_grid=False)


def test_greedy_takes_immediate_win_and_avoids_immediate_loss():
    rng = np.random.default_rng(1)
    greedy = GreedyOpponent()
    checked = 0
    for seed in range(60):
        g = PaperSoccerGame(4, 4)
        g.reset()
        r2 = np.random.default_rng(seed)
        end = False
        while not end:
            me = g.current_player
            outcomes = {int(a): g.peek(int(a)) for a in g.legal_actions()}
            wins = [a for a, (_, r, e) in outcomes.items() if e and r[me] == 1]
            safe = [a for a, (_, r, e) in outcomes.items() if not (e and r[me] == -1)]
            a = greedy.act(g, rng)
            if wins:
                assert a in wins
                checked += 1
            elif safe:
                assert a in safe
            _, _, end, _ = g.step(int(r2.choice(g.legal_actions())), copy_grid=False)
    assert checked > 0


def test_greedy_beats_random():
    rows = evaluate_vs(GreedyOpponent(), RandomOpponent(), (4, 4), n_games=100, seed=0)
    assert all(r["win_rate"] > 0.7 for r in rows), rows


def test_play_game_is_reproducible():
    g = PaperSoccerGame(6, 8)
    a = play_game({-1: GreedyOpponent(), 1: RandomOpponent()}, g, np.random.default_rng(5))
    b = play_game({-1: GreedyOpponent(), 1: RandomOpponent()}, g, np.random.default_rng(5))
    assert a == b


def test_tabular_agent_only_legal_and_learns_something():
    agent = TabularQAgent()
    g = PaperSoccerGame(4, 4)
    rng = np.random.default_rng(0)
    for ep in range(3000):
        agent.train_episode(g, RandomOpponent(), -1 if ep % 2 == 0 else 1, rng, eps=0.3)
    assert agent.n_states > 100
    # polityka zachłanna: zawsze legalne akcje
    for _ in range(50):
        res = play_game({-1: agent, 1: RandomOpponent()}, g, rng)
        assert res["winner"] in (-1, 1)
    rows = evaluate_vs(agent, RandomOpponent(), (4, 4), n_games=200, seed=1)
    assert np.mean([r["win_rate"] for r in rows]) > 0.55  # powyżej losowego (0.5)
