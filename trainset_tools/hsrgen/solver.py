"""Small backend-independent linear/MIP builder; no pandas dependency."""
import math
import time


class LinearModel:
    def __init__(self):
        self.lb, self.ub, self.integer, self.cost = [], [], [], []
        self.rows, self.lower, self.upper, self.names = [], [], [], []

    def var(self, lower=0, upper=math.inf, integer=False, cost=0):
        i = len(self.lb)
        self.lb.append(lower); self.ub.append(upper)
        self.integer.append(int(integer)); self.cost.append(cost)
        return i

    def add(self, values, lower=-math.inf, upper=math.inf, name=""):
        self.rows.append({i: float(v) for i, v in values.items() if abs(v) > 1e-12})
        self.lower.append(lower); self.upper.append(upper); self.names.append(name)

    def absolute(self, values, target, weight):
        z = self.var(cost=weight)
        a = dict(values); a[z] = -1
        self.add(a, upper=target)
        a = {i: -v for i, v in values.items()}; a[z] = -1
        self.add(a, upper=-target)

    def exclude_integer_vector(self, variables, previous):
        flags = []
        for v, p in zip(variables, previous):
            if p < self.ub[v]:
                b = self.var(0, 1, True); flags.append(b)
                big = p + 1 - self.lb[v]
                self.add({v: 1, b: -big}, lower=p + 1 - big)
            if p > self.lb[v]:
                b = self.var(0, 1, True); flags.append(b)
                big = self.ub[v] - p + 1
                self.add({v: 1, b: big}, upper=p - 1 + big)
        self.add({b: 1 for b in flags}, lower=1, name="distinct_composition")

    def solve(self, backend="auto", seconds=60, seed=40, threads=1, log=False, gap=0.05):
        start = time.perf_counter()
        fallback = ""
        if backend in ("auto", "gurobi"):
            try:
                import gurobipy as gp
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", int(log)); env.start()
                try:
                    with gp.Model(env=env) as m:
                        m.Params.TimeLimit = seconds
                        m.Params.Seed = int(seed) % 2000000000
                        m.Params.Threads = threads
                        m.Params.MIPGap = gap
                        vs = [m.addVar(lb=l, ub=u, obj=c, vtype="I" if t else "C")
                              for l, u, c, t in zip(self.lb, self.ub, self.cost, self.integer)]
                        for a, l, u, name in zip(self.rows, self.lower, self.upper, self.names):
                            e = gp.LinExpr(list(a.values()), [vs[i] for i in a])
                            if l == u: m.addConstr(e == l, name=name)
                            else:
                                if math.isfinite(l): m.addConstr(e >= l, name=name + "_lo")
                                if math.isfinite(u): m.addConstr(e <= u, name=name + "_hi")
                        m.optimize()
                        x = [v.X for v in vs] if m.SolCount else None
                        status = ("OPTIMAL" if m.Status == gp.GRB.OPTIMAL else
                                  "FEASIBLE_LIMIT" if x is not None else
                                  "INFEASIBLE" if m.Status == gp.GRB.INFEASIBLE else "UNKNOWN_LIMIT")
                        return self._result(x, status, "gurobi", start, str(m.Status))
                finally:
                    env.dispose()
            except (ImportError, Exception) as exc:
                if backend == "gurobi": raise RuntimeError("Gurobi unavailable: " + str(exc)) from exc
                fallback = str(exc)
        if backend not in ("auto", "scipy"):
            raise ValueError("Unknown solver: " + backend)
        try:
            import numpy as np
            from scipy.optimize import milp, Bounds, LinearConstraint
            from scipy.sparse import coo_matrix
        except ImportError as exc:
            raise RuntimeError("Install scipy>=1.9 or use a licensed gurobipy. " + fallback) from exc
        ri, ci, vv = [], [], []
        for j, row in enumerate(self.rows):
            for i, v in row.items(): ri.append(j); ci.append(i); vv.append(v)
        a = coo_matrix((vv, (ri, ci)), shape=(len(self.rows), len(self.lb))).tocsc()
        # scipy's documented milp options do not expose thread count.
        r = milp(np.array(self.cost), integrality=np.array(self.integer),
                 bounds=Bounds(self.lb, self.ub),
                 constraints=LinearConstraint(a, self.lower, self.upper),
                 options={"time_limit": seconds, "mip_rel_gap": gap, "disp": log})
        x = r.x.tolist() if r.x is not None else None
        status = ("OPTIMAL" if r.status == 0 else "FEASIBLE_LIMIT" if x is not None
                  else "INFEASIBLE" if r.status == 2 else "UNKNOWN_LIMIT")
        result = self._result(x, status, "scipy", start, str(r.message))
        if fallback: result["backend_fallback_reason"] = fallback
        return result

    def _result(self, x, status, backend, start, message):
        if x is not None:
            for i, integer in enumerate(self.integer):
                if integer:
                    if abs(x[i] - round(x[i])) > 1e-5: raise RuntimeError("Nonintegral incumbent")
                    x[i] = round(x[i])
            # Validate solver incumbent rather than trusting the status alone.
            for i, v in enumerate(x):
                if v < self.lb[i] - 1e-5 or v > self.ub[i] + 1e-5:
                    raise RuntimeError("Variable bound violation")
            for a, l, u in zip(self.rows, self.lower, self.upper):
                v = sum(c * x[i] for i, c in a.items())
                if v < l - 1e-5 or v > u + 1e-5: raise RuntimeError("Constraint violation")
        return {"x": x, "status": status, "backend": backend,
                "seconds": time.perf_counter() - start, "variables": len(self.lb),
                "constraints": len(self.rows), "message": message}
