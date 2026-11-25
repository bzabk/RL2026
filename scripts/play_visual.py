#!/usr/bin/env python3
"""Interactive, offline replay of a Paper Soccer game.

Place in scripts/ in the RL2026 repository and run from its root.
Examples:
  uv run python scripts/play_visual.py --board 8x10
  uv run python scripts/play_visual.py --board 8x10 --agent results/tabular_q_8x10_s0/final.pkl
  uv run python scripts/play_visual.py --board 4x4 --player1 greedy --player2 random
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_soccer.agents.tabular_q import TabularQAgent
from paper_soccer.game import ACTION_MAP, PaperSoccerGame, parse_board
from paper_soccer.opponents import GreedyOpponent, RandomOpponent
from paper_soccer.utils.io import load_pickle


def load_player(spec: str, board: tuple[int, int], gate_size: int):
    """Return an agent and a readable label. Load only trusted local pickles."""
    if spec == "random":
        return RandomOpponent(), "Random"
    if spec == "greedy":
        return GreedyOpponent(), "Greedy"
    path = Path(spec)
    if not path.is_file():
        raise ValueError(f"Model file not found: {path}")
    config_path = path.parent / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "board" in config and parse_board(config["board"]) != board:
            raise ValueError(f"{path}: trained on {config['board']}, not {board[0]}x{board[1]}")
        if "gate_size" in config and config["gate_size"] != gate_size:
            raise ValueError(f"{path}: model uses gate size {config['gate_size']}")
        if config.get("algo", "tabular_q") != "tabular_q":
            raise ValueError("This version supports Tabular Q-learning checkpoints only.")
    state = load_pickle(path)
    if isinstance(state, dict) and "agent" in state:
        state = state["agent"]  # ckpt.pkl, rather than final.pkl / milestone_*.pkl
    if not isinstance(state, dict) or "q" not in state:
        raise ValueError(f"{path}: not a supported Tabular Q-learning checkpoint")
    agent = TabularQAgent()
    agent.load_state_dict(state)
    return agent, f"Q-learning · {path.parent.name}/{path.name}"


def xy(pos):
    """Engine positions are (row, column); the replay uses (x, y)."""
    return [int(pos[1]), int(pos[0])]


def snapshot(game):
    y, x = game.cur_pos
    legal = [] if game.done else [
        [x + ACTION_MAP[int(a)][1], y + ACTION_MAP[int(a)][0]]
        for a in game.legal_actions()
    ]
    return {"ball": xy(game.cur_pos), "next": int(game.current_player),
            "legal": legal, "done": bool(game.done), "winner": int(game.winner),
            "reason": game.termination_reason}


def record_game(game, players, labels, seed: int):
    """Run one game once; replaying it never calls agents or changes their state."""
    game.reset()
    rng = np.random.default_rng(seed)
    initial = snapshot(game)
    moves = []
    while not game.done:
        actor = game.current_player
        start = xy(game.cur_pos)
        agent = players[actor]
        known = None
        if isinstance(agent, TabularQAgent):
            known = game.state_key() in agent.q
        action = int(agent.act(game, rng, deterministic=True))
        _, _, done, nxt = game.step(action, copy_grid=False)
        moves.append({"actor": int(actor), "start": start, "end": xy(game.cur_pos),
                      "action": action, "bounce": bool(not done and nxt == actor),
                      "known": known, **snapshot(game)})
    return {"width": game.x_length, "height": game.y_length, "gate": game.gate_size,
            "seed": seed, "labels": labels, "initial": initial, "moves": moves}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Soccer | Match replay</title>
<style>
:root{color-scheme:light;--blue:#2563eb;--red:#dc2626;--ink:#172334;--muted:#64748b;--line:#dce4ec}
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1060px;margin:28px auto;padding:0 18px}h1{font-size:28px;letter-spacing:-.8px;margin:0}h2{font-size:16px;margin:0 0 10px}p{margin:5px 0}header{margin-bottom:18px}.muted{color:var(--muted);font-size:13px}.layout{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,.85fr);gap:16px;align-items:start}.panel{background:#fff;border:1px solid var(--line);border-radius:15px;padding:16px}.pitch{display:block;width:100%;height:min(64vh,650px);min-height:280px}.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}button,select{font:inherit;border:1px solid #cbd5e1;border-radius:7px;background:white;color:var(--ink);padding:7px 10px;cursor:pointer}button:disabled{opacity:.4;cursor:default}button.primary{background:#172334;color:white;border-color:#172334}input[type=range]{width:100%;accent-color:#2563eb}.legend{display:flex;flex-wrap:wrap;gap:13px;font-size:12px;color:#475569}.swatch{display:inline-block;width:16px;height:4px;border-radius:3px;vertical-align:middle;margin-right:5px}.player{border-left:4px solid;padding:8px 10px;background:#f8fafc;border-radius:5px;margin:8px 0;overflow-wrap:anywhere}.player strong{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.06em}.status{padding:12px;background:#f1f5f9;border-radius:8px;margin:12px 0}.status strong{display:block;font-size:17px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}.stat{border:1px solid var(--line);border-radius:8px;padding:8px 10px}.stat b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.history{max-height:270px;overflow:auto;border-top:1px solid var(--line)}.event{width:100%;text-align:left;border:0;border-bottom:1px solid #edf1f5;border-radius:0;padding:7px 4px;font-size:12px;display:flex;gap:8px;align-items:center;background:white}.event.active{background:#eaf0fa;font-weight:650}.event.future{opacity:.45}.dot{height:9px;width:9px;border-radius:50%;flex-shrink:0}.check{display:flex;align-items:center;gap:6px;font-size:13px}@media(max-width:740px){.layout{grid-template-columns:1fr}.pitch{height:56vh}main{margin:14px auto}.history{max-height:180px}}
</style></head><body><main>
<header><h1>Paper Soccer</h1><p id="subtitle" class="muted"></p></header>
<div class="layout"><section class="panel"><svg id="pitch" class="pitch" role="img" aria-label="Paper Soccer board" xmlns="http://www.w3.org/2000/svg"></svg>
<div class="legend"><span><i class="swatch" style="background:#2563eb"></i>Player 1</span><span><i class="swatch" style="background:#dc2626"></i>Player 2</span><span>◉ Ball</span><span>Dashed = legal move</span></div>
<div class="controls"><button id="restart">Start</button><button id="back">◀ Previous</button><button id="play" class="primary">Play</button><button id="forward">Next ▶</button><button id="end">End</button></div>
<input type="range" id="seek" min="0" value="0" aria-label="Replay position"><div class="controls" style="justify-content:space-between"><span id="counter" class="muted"></span><label class="check">Speed <select id="speed"><option value="1200">0.5×</option><option value="600" selected>1×</option><option value="300">2×</option><option value="150">4×</option></select></label><label class="check"><input type="checkbox" id="legal" checked> Legal moves</label></div>
<p class="muted">Keyboard: Space = play / pause, ← / → = previous / next.</p></section>
<aside class="panel"><h2>Match details</h2><div id="players"></div><div class="status" aria-live="polite"><strong id="status"></strong><span id="detail" class="muted"></span></div><div class="stats"><div class="stat"><span class="muted">Player 1 moves</span><b id="m1">0</b></div><div class="stat"><span class="muted">Player 2 moves</span><b id="m2">0</b></div><div class="stat"><span class="muted">Bounces</span><b id="bounces">0</b></div><div class="stat"><span class="muted">Legal moves now</span><b id="nlegal">0</b></div></div><p id="knowledge" class="muted"></p><h2 style="margin-top:16px">Move history</h2><div id="history" class="history"></div></aside></div></main>
<script type="application/json" id="match-data">__MATCH_DATA__</script>
<script>
const D=JSON.parse(document.getElementById('match-data').textContent);
const $=id=>document.getElementById(id), svg=$('pitch'), ns='http://www.w3.org/2000/svg';
const colors={'-1':'#2563eb','1':'#dc2626'}, names={'-1':'Player 1','1':'Player 2'};
const N=D.moves.length, W=D.width, bottom=D.height+2, L=(W-D.gate)/2,R=(W+D.gate)/2;
let step=0,timer=null;
function el(tag,attrs,parent=svg){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e}
function line(a,b,color,width,extra={}){return el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:color,'stroke-width':width,'stroke-linecap':'round',...extra})}
function circle(p,r,fill,extra={}){return el('circle',{cx:p[0],cy:p[1],r,fill,...extra})}
function textAt(x,y,t,fill){const e=el('text',{x,y,'font-size':.31,'text-anchor':'middle',fill,'font-weight':650});e.textContent=t}
function board(){
 svg.replaceChildren();svg.setAttribute('viewBox',`${-0.65} ${-0.8} ${W+1.3} ${bottom+1.6}`);
 const points=[[0,1],[L,1],[L,0],[R,0],[R,1],[W,1],[W,bottom-1],[R,bottom-1],[R,bottom],[L,bottom],[L,bottom-1],[0,bottom-1]];
 el('polygon',{points:points.map(p=>p.join(',')).join(' '),fill:'#f0f8f2',stroke:'#64748b','stroke-width':.065,'stroke-linejoin':'round'});
 for(let y=1;y<bottom;y++)for(let x=0;x<=W;x++)circle([x,y],.032,'#a7b9ab');
 for(const y of [0,bottom])for(let x=L;x<=R;x++)circle([x,y],.045,'#a7b9ab');
 line([0,bottom/2],[W,bottom/2],'#bed1c2',.025,{'stroke-dasharray':'.10 .12'});
 circle([W/2,bottom/2],.55,'none',{stroke:'#bed1c2','stroke-width':.025});
 line([L,0],[R,0],colors[1],.085);line([L,bottom],[R,bottom],colors[-1],.085);
 textAt(W/2,-.28,'Player 2 target ↑',colors[1]);textAt(W/2,bottom+.48,'Player 1 target ↓',colors[-1]);
}
function current(){return step===0?D.initial:D.moves[step-1]}
function render(){
 board();const s=current();
 const used=D.moves.slice(0,step);
 used.forEach((m,i)=>{if(m.start[0]!==m.end[0]||m.start[1]!==m.end[1])line(m.start,m.end,colors[m.actor],i===step-1?.105:.065,{opacity:i===step-1?1:.76});if(m.bounce)circle(m.end,.065,'#fff',{stroke:colors[m.actor],'stroke-width':.045});});
 if($('legal').checked&&!s.done)s.legal.forEach(p=>{line(s.ball,p,colors[s.next],.036,{'stroke-dasharray':'.11 .10',opacity:.65});circle(p,.072,'#fff',{stroke:colors[s.next],'stroke-width':.03});});
 circle(s.ball,.145,'#fff',{stroke:'#172334','stroke-width':.055});circle(s.ball,.062,'#172334');
 $('counter').textContent=`Move ${step} / ${N}`;$('seek').value=step;
 $('m1').textContent=used.filter(m=>m.actor===-1).length;$('m2').textContent=used.filter(m=>m.actor===1).length;$('bounces').textContent=used.filter(m=>m.bounce).length;$('nlegal').textContent=s.legal.length;
 const last=used.at(-1);const reason={goal_top:'Goal at the upper end',goal_bottom:'Goal at the lower end',blocked:'Ball blocked',illegal_move:'Illegal move'};
 if(s.done){$('status').textContent=`${names[s.winner]} wins`; $('status').style.color=colors[s.winner]||'#172334';$('detail').textContent=reason[s.reason]||s.reason||'Game over'}
 else{$('status').textContent=`${names[s.next]} to move`; $('status').style.color=colors[s.next];$('detail').textContent=last?(last.bounce?`${names[last.actor]} bounced — same player moves again.`:`${names[last.actor]} moved — turn changes.`):'Ready to start.'}
 $('knowledge').textContent=last&&last.known!==null?(last.known?'Last Q-learning move: state found in Q-table (ties may still be random).':'Last Q-learning move: unseen state → random legal action.') : '';
 $('back').disabled=step===0;$('forward').disabled=step===N;$('restart').disabled=step===0;$('end').disabled=step===N;
 for(const b of $('history').children){const i=Number(b.dataset.step);b.classList.toggle('active',i===step);b.classList.toggle('future',i>step);if(i===step)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current')}
 const active=$('history').querySelector('.active');if(active){const h=$('history');if(active.offsetTop<h.offsetTop+h.scrollTop||active.offsetTop+active.offsetHeight>h.offsetTop+h.scrollTop+h.clientHeight)h.scrollTop=active.offsetTop-h.offsetTop-h.clientHeight/2;}
 if(step===N&&timer!==null)pause();
}
function pause(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='Play'}
function start(){if(step===N)step=0;render();timer=setInterval(()=>{if(step<N){step++;render()}else pause()},Number($('speed').value));$('play').textContent='Pause'}
function jump(n){pause();step=Math.max(0,Math.min(N,n));render()}
$('subtitle').textContent=`${W} × ${D.height} board · seed ${D.seed} · ${N} recorded moves`;
$('seek').max=N;
for(const side of [-1,1]){const div=document.createElement('div');div.className='player';div.style.borderColor=colors[side];const title=document.createElement('strong');title.textContent=names[side]+(side===-1?' · starts · attacks down':' · attacks up');title.style.color=colors[side];div.append(title,document.createTextNode(D.labels[side]));$('players').append(div)}
D.moves.forEach((m,i)=>{const b=document.createElement('button');b.className='event';b.dataset.step=i+1;const dot=document.createElement('span');dot.className='dot';dot.style.background=colors[m.actor];const directions=['NW','N','NE','E','SE','S','SW','W'];b.append(dot,document.createTextNode(`${i+1}. ${names[m.actor]} · ${directions[m.action]||'?'}${m.bounce?' · bounce':''}${m.done?' · finish':''}`));b.onclick=()=>jump(i+1);$('history').append(b)});
$('restart').onclick=()=>jump(0);$('back').onclick=()=>jump(step-1);$('forward').onclick=()=>jump(step+1);$('end').onclick=()=>jump(N);$('play').onclick=()=>timer===null?start():pause();$('seek').oninput=e=>jump(Number(e.target.value));$('legal').onchange=render;$('speed').onchange=()=>{if(timer!==null){pause();start()}};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click()}else if(e.key==='ArrowLeft'){e.preventDefault();jump(step-1)}else if(e.key==='ArrowRight'){e.preventDefault();jump(step+1)}});
render();
</script></body></html>'''


def write_replay(data, path: Path):
    # Prevent arbitrary player labels from breaking out of the embedded JSON script.
    encoded = json.dumps(data, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HTML.replace("__MATCH_DATA__", encoded), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board", default="4x4")
    p.add_argument("--gate-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--player1", default=None, help="random, greedy, or a trusted model .pkl; starts and attacks down")
    p.add_argument("--player2", default="random", help="random, greedy, or a trusted model .pkl; attacks up")
    p.add_argument("--agent", help="shortcut for --player1 MODEL (compatible with play_demo.py)")
    p.add_argument("--output", type=Path, help="output HTML path (default: results/replays/replay_<board>_s<seed>.html)")
    p.add_argument("--no-open", action="store_true", help="save the HTML without opening a browser")
    a = p.parse_args()
    if a.agent and a.player1:
        p.error("Use either --agent or --player1, not both.")
    try:
        board = parse_board(a.board)
        if min(board) < 2 or a.gate_size <= 0 or a.gate_size >= min(board):
            raise ValueError("Use an even positive board and a positive gate smaller than both dimensions.")
        game = PaperSoccerGame(*board, a.gate_size)
        first, label1 = load_player(a.agent or a.player1 or "greedy", board, a.gate_size)
        second, label2 = load_player(a.player2, board, a.gate_size)
        data = record_game(game, {-1: first, 1: second}, {-1: label1, 1: label2}, a.seed)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    path = (a.output or Path(f"results/replays/replay_{board[0]}x{board[1]}_s{a.seed}.html")).resolve()
    write_replay(data, path)
    print(f"Replay: {path}")
    print(f"Moves: {len(data['moves'])} | Winner: {game.winner} | {game.termination_reason}")
    if not a.no_open:
        if not webbrowser.open(path.as_uri()):
            print("Open the HTML file manually in a browser.")


if __name__ == "__main__":
    main()
