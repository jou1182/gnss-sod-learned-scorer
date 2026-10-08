"""Train the learned removal policy: behaviour-cloning warm start, then REINFORCE (POMO-style
shared baseline). Rollouts and gradient computation run in parallel worker processes."""
import sys, time, json, os
import numpy as np
import torch
from multiprocessing import Pool

from sod.env import synth_corridor
from sod.baselines import backward
from sod.bdp import BackwardEnv
from sod.policy import Scorer, rollout

TRAIN0, VAL0 = 10_000, 20_000
_model = None


def _init():
    global _model
    torch.set_num_threads(1)
    _model = Scorer()


def _load(sd):
    _model.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})


def _grads():
    return [p.grad.numpy().copy() if p.grad is not None else np.zeros(p.shape, np.float32) for p in _model.parameters()]


def _bc_task(args):
    sd, seed, rule = args
    _load(sd); _model.zero_grad(set_to_none=False)
    inst = synth_corridor(seed)
    env = BackwardEnv(inst)
    losses = []
    n = 0
    while True:
        gi, F = env.features()
        if len(gi) == 0:
            break
        # expert action
        if rule == "longest":
            a = int(np.argmax(inst.cost[gi]))
        else:
            a = int(np.argmax(F[:, 12] * 0 + inst.cost[gi] / (2 - env.slack["h"][0][env.feas] - env.slack["u"][0][env.feas] + 1e-9)))
        logits = _model(torch.from_numpy(F))
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([a]))
        loss.backward()
        losses.append(loss.item()); n += 1
        env.remove(int(gi[a]))
    for p in _model.parameters():
        p.grad /= max(n, 1)
    return _grads(), float(np.mean(losses)) if losses else 0.0


def _rl_task(args):
    sd, seed, K, ent = args
    _load(sd); _model.zero_grad(set_to_none=False)
    rng = np.random.default_rng(seed)
    inst = synth_corridor(seed)
    ref = backward(inst, "ratio")["cost"]
    trajs, R = [], []
    for _ in range(K):
        env, tr = rollout(inst, _model, "sample", rng, store=True)
        trajs.append(tr); R.append(-env.cost() / ref)
    R = np.array(R); adv = (R - R.mean()) / (R.std() + 1e-3)
    tot = 0.0
    for tr, A in zip(trajs, adv):
        loss = 0.0
        for F, a in tr:
            lg = _model(torch.from_numpy(F))
            lp = torch.log_softmax(lg, 0)
            loss = loss - A * lp[a] - ent * (-(torch.softmax(lg, 0) * lp).sum())
        (loss / (len(trajs) * 100.0)).backward()
    return _grads(), float(R.mean()), float(R.max())


def _eval_task(args):
    sd, seed = args
    _load(sd)
    inst = synth_corridor(seed)
    ref = backward(inst, "ratio")["cost"]
    env, _ = rollout(inst, _model, "greedy")
    return env.cost() / ref


def get_sd(m):
    return {k: v.detach().numpy().copy() for k, v in m.state_dict().items()}


def apply(model, opt, grad_lists):
    g = [np.mean([gl[i] for gl in grad_lists], 0) for i in range(len(grad_lists[0]))]
    for p, gi in zip(model.parameters(), g):
        p.grad = torch.from_numpy(gi.astype(np.float32))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def main(bc_iters=40, rl_iters=300, workers=28, K=6, out="results/policy.pt"):
    torch.manual_seed(0)
    model = Scorer()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    log = []
    with Pool(workers, initializer=_init) as pool:
        t0 = time.time()
        for it in range(bc_iters):
            sd = get_sd(model)
            res = pool.map(_bc_task, [(sd, TRAIN0 + it * workers + j, "ratio") for j in range(workers)])
            apply(model, opt, [r[0] for r in res])
            if it % 10 == 0:
                print(f"BC {it} loss {np.mean([r[1] for r in res]):.3f} t={time.time()-t0:.0f}s", flush=True)
        for g in opt.param_groups: g["lr"] = 5e-4
        best = 1e9
        for it in range(rl_iters):
            sd = get_sd(model)
            res = pool.map(_rl_task, [(sd, TRAIN0 + 5000 + it * workers + j, K, 0.003) for j in range(workers)])
            apply(model, opt, [r[0] for r in res])
            if it % 10 == 0:
                ev = pool.map(_eval_task, [(get_sd(model), VAL0 + j) for j in range(56)])
                v = float(np.mean(ev))
                log.append(dict(it=it, train_ratio=-float(np.mean([r[1] for r in res])), val_ratio=v))
                print(f"RL {it} train_cost/ref {-np.mean([r[1] for r in res]):.4f} val greedy/ref {v:.4f} t={time.time()-t0:.0f}s", flush=True)
                if v < best:
                    best = v; torch.save(model.state_dict(), out)
        torch.save(model.state_dict(), out.replace(".pt", "_last.pt"))
    json.dump(log, open("results/train_log.json", "w"))


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main(*(int(a) for a in sys.argv[1:3])) if len(sys.argv) > 2 else main()
