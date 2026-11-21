"""Testy poprawności silnika (blokują dalszy trening - plan, sekcja 4.4)."""
import numpy as np
import pytest

from paper_soccer.game import ACTION_MAP, PaperSoccerGame

BOARDS = [(4, 4), (6, 8), (8, 10)]


def random_game(board, seed):
    rng = np.random.default_rng(seed)
    g = PaperSoccerGame(*board)
    g.reset()
    return g, rng


@pytest.mark.parametrize("board", BOARDS)
def test_moves_are_legal_unit_steps_and_edges_never_return(board):
    for seed in range(30):
        g, rng = random_game(board, seed)
        used = set()
        end = False
        while not end:
            legal = g.legal_actions()
            assert len(legal) > 0
            a = int(rng.choice(legal))
            y, x = g.cur_pos
            dy, dx = ACTION_MAP[a]
            edge = frozenset([(y, x), (y + dy, x + dx)])
            assert edge not in used, "zużyta krawędź wróciła do legalnych"
            used.add(edge)
            _, _, end, _ = g.step(a)
            assert g.cur_pos == [y + dy, x + dx]


@pytest.mark.parametrize("board", BOARDS)
def test_bounce_keeps_player_otherwise_switches(board):
    seen_bounce = seen_switch = False
    for seed in range(30):
        g, rng = random_game(board, seed)
        end = False
        while not end:
            a = int(rng.choice(g.legal_actions()))
            who = g.current_player
            y, x = g.cur_pos
            dy, dx = ACTION_MAP[a]
            was_marked = g.grid[y + dy, x + dx, 8] == 1  # punkt brzegowy/odwiedzony => odbicie
            _, _, end, nxt = g.step(a)
            if end:
                break
            if was_marked:
                assert nxt == who
                seen_bounce = True
            else:
                assert nxt == -who
                seen_switch = True
    assert seen_bounce and seen_switch


@pytest.mark.parametrize("board", BOARDS)
def test_terminal_rewards_zero_sum_and_finite_games(board):
    reasons = set()
    for seed in range(60):
        g, rng = random_game(board, seed)
        end, n = False, 0
        while not end:
            _, rewards, end, _ = g.step(int(rng.choice(g.legal_actions())))
            n += 1
            assert n < 2000
        assert sorted(rewards.values()) == [-1, 1]
        assert g.winner in (-1, 1) and rewards[g.winner] == 1
        reasons.add(g.termination_reason)
    assert reasons <= {"goal_top", "goal_bottom", "blocked", "illegal_move"}
    assert "goal_top" in reasons or "goal_bottom" in reasons


def test_illegal_move_loses():
    g, _ = random_game((4, 4), 0)
    y, x = g.cur_pos
    a = int(g.legal_actions()[0])
    g.step(a)
    # wróć po tej samej krawędzi (zużyta) -> niedostępna
    back = (a - 4) % 8
    mover = g.current_player
    _, rewards, end, _ = g.step(back)
    assert end and rewards[mover] == -1 and g.termination_reason == "illegal_move"


@pytest.mark.parametrize("board", BOARDS)
def test_get_set_state_roundtrip_and_copy_independence(board):
    g, rng = random_game(board, 1)
    for _ in range(6):
        g.step(int(rng.choice(g.legal_actions())))
    state = g.get_state()
    snap = g.copy()
    g.step(int(rng.choice(g.legal_actions())))
    assert snap.cur_pos != g.cur_pos or not np.array_equal(snap.grid, g.grid)  # kopia niezależna
    g.set_state(state)
    assert np.array_equal(g.grid, snap.grid)
    assert g.cur_pos == snap.cur_pos and g.current_player == snap.current_player
    assert np.array_equal(g.action_masks(), snap.action_masks())
    assert g.state_key() == snap.state_key()


@pytest.mark.parametrize("board", BOARDS)
def test_peek_equals_step_and_does_not_mutate(board):
    """Wymóg Dyna: simulate(copy(s), a) == step(s, a), oryginał nietknięty."""
    for seed in range(10):
        g, rng = random_game(board, seed)
        end = False
        while not end:
            before = g.get_state()
            for a in g.legal_actions():
                sim, r_sim, e_sim = g.peek(int(a))
                assert np.array_equal(g.grid, before[0]) and g.cur_pos == list(before[1])  # bez mutacji
                real = g.copy()
                grid, r_real, e_real, p_real = real.step(int(a))
                assert r_sim == r_real and e_sim == e_real
                assert np.array_equal(sim.grid, grid) and sim.cur_pos == real.cur_pos
                assert sim.current_player == p_real
                if not e_real:
                    assert np.array_equal(sim.action_masks(), real.action_masks())
            _, _, end, _ = g.step(int(rng.choice(g.legal_actions())))


def test_state_key_distinguishes_states():
    g, rng = random_game((4, 4), 3)
    keys = {g.state_key()}
    for _ in range(5):
        g.step(int(rng.choice(g.legal_actions())))
        keys.add(g.state_key())
    assert len(keys) == 6


def test_render_runs():
    g, rng = random_game((4, 4), 2)
    for _ in range(5):
        g.step(int(rng.choice(g.legal_actions())))
    s = g.render()
    assert "O" in s and "\n" in s
