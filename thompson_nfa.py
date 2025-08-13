#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador y simulador de AFN usando el algoritmo de Thompson.
Lee expresiones regulares desde un archivo, construye el AFN y lo dibuja en PNG.
También permite simular cadenas para verificar pertenencia al lenguaje.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt

EPS = 'ε'

@dataclass
class NFA:
    start: int
    accepts: Set[int]
    transitions: Dict[int, List[Tuple[Optional[str], int]]]

    def add_transition(self, src: int, symbol: Optional[str], dst: int):
        self.transitions.setdefault(src, []).append((symbol, dst))

@dataclass
class Fragment:
    start: int
    accepts: Set[int]

class ThompsonBuilder:
    def __init__(self):
        self.state_counter = 0
        self.nfa = NFA(start=-1, accepts=set(), transitions={})

    def new_state(self) -> int:
        sid = self.state_counter
        self.state_counter += 1
        self.nfa.transitions.setdefault(sid, [])
        return sid

    def symbol(self, c: str) -> Fragment:
        s = self.new_state()
        f = self.new_state()
        self.nfa.add_transition(s, c, f)
        return Fragment(s, {f})

    def concat(self, a: Fragment, b: Fragment) -> Fragment:
        for acc in a.accepts:
            self.nfa.add_transition(acc, EPS, b.start)
        return Fragment(a.start, b.accepts)

    def union(self, a: Fragment, b: Fragment) -> Fragment:
        s = self.new_state()
        f = self.new_state()
        self.nfa.add_transition(s, EPS, a.start)
        self.nfa.add_transition(s, EPS, b.start)
        for acc in a.accepts:
            self.nfa.add_transition(acc, EPS, f)
        for acc in b.accepts:
            self.nfa.add_transition(acc, EPS, f)
        return Fragment(s, {f})

    def star(self, a: Fragment) -> Fragment:
        s = self.new_state()
        f = self.new_state()
        self.nfa.add_transition(s, EPS, a.start)
        self.nfa.add_transition(s, EPS, f)
        for acc in a.accepts:
            self.nfa.add_transition(acc, EPS, a.start)
            self.nfa.add_transition(acc, EPS, f)
        return Fragment(s, {f})

    def question(self, a: Fragment) -> Fragment:
        s = self.new_state()
        f = self.new_state()
        self.nfa.add_transition(s, EPS, a.start)
        self.nfa.add_transition(s, EPS, f)
        for acc in a.accepts:
            self.nfa.add_transition(acc, EPS, f)
        return Fragment(s, {f})

def regex_to_postfix(regex: str) -> str:
    def is_symbol(ch):
        return ch.isalnum() or ch == EPS
    clean = "".join(regex.split())
    result = []
    for i, c in enumerate(clean):
        result.append(c)
        if i == len(clean)-1:
            break
        d = clean[i+1]
        if (c in [')','*','+','?'] or c.isalnum() or c==EPS) and (d.isalnum() or d==EPS or d=='('):
            result.append('.')
    infix = "".join(result)
    prec = {'|':1, '.':2, '*':3, '+':3, '?':3}
    right_assoc = {'*', '+', '?'}
    output = []
    stack = []
    i = 0
    while i < len(infix):
        c = infix[i]
        if c == '(':
            stack.append(c)
        elif c == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        elif c in prec:
            while stack and stack[-1] != '(' and ((prec[stack[-1]] > prec[c]) or (prec[stack[-1]] == prec[c] and c not in right_assoc)):
                output.append(stack.pop())
            stack.append(c)
        else:
            output.append(c)
        i += 1
    while stack:
        output.append(stack.pop())
    return " ".join(output)

def build_nfa_from_postfix(postfix: str) -> NFA:
    tb = ThompsonBuilder()
    stack: List[Fragment] = []
    tokens = postfix.split()
    for t in tokens:
        if t in {'|', '.', '*', '+', '?'}:
            if t == '*':
                a = stack.pop()
                stack.append(tb.star(a))
            elif t == '+':
                a = stack.pop()
                a_copy = tb.star(a)  # plus como a seguido de a*
                stack.append(tb.concat(a, a_copy))
            elif t == '?':
                a = stack.pop()
                stack.append(tb.question(a))
            elif t == '.':
                b = stack.pop()
                a = stack.pop()
                stack.append(tb.concat(a, b))
            elif t == '|':
                b = stack.pop()
                a = stack.pop()
                stack.append(tb.union(a, b))
        else:
            stack.append(tb.symbol(t))
    frag = stack.pop()
    tb.nfa.start = frag.start
    tb.nfa.accepts = frag.accepts
    return tb.nfa

def epsilon_closure(nfa: NFA, states: Set[int]) -> Set[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for sym, dst in nfa.transitions.get(s, []):
            if sym == EPS or sym is None:
                if dst not in closure:
                    closure.add(dst)
                    stack.append(dst)
    return closure

def move(nfa: NFA, states: Set[int], symbol: str) -> Set[int]:
    dest = set()
    for s in states:
        for sym, dst in nfa.transitions.get(s, []):
            if sym == symbol:
                dest.add(dst)
    return dest

def nfa_accepts(nfa: NFA, w: str) -> bool:
    current = epsilon_closure(nfa, {nfa.start})
    for ch in w:
        current = epsilon_closure(nfa, move(nfa, current, ch))
    return any(s in nfa.accepts for s in current)

def draw_nfa(nfa: NFA, path: str, title: str = "AFN"):
    G = nx.DiGraph()
    edge_labels = {}
    for src, edges in nfa.transitions.items():
        for sym, dst in edges:
            label = EPS if (sym == EPS or sym is None) else sym
            key = (src, dst)
            edge_labels.setdefault(key, [])
            edge_labels[key].append(label)
            G.add_edge(src, dst)
    labels = {}
    for node in nfa.transitions.keys() | nfa.accepts | {nfa.start}:
        suffix = []
        if node == nfa.start:
            suffix.append("S")
        if node in nfa.accepts:
            suffix.append("A")
        labels[node] = f"{node}" + (f" ({','.join(suffix)})" if suffix else "")
        G.add_node(node)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=12, connectionstyle='arc3,rad=0.15')
    merged = {k: ",".join(sorted(set(v))) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v): lab for (u,v), lab in merged.items()}, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generador y simulador de AFN por Thompson")
    parser.add_argument("--file", required=True, help="Archivo con expresiones regulares (una por línea)")
    parser.add_argument("--w", help="Cadena a evaluar (opcional, si no se da, solo genera AFN)")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            regex = line.strip()
            if not regex:
                continue
            pf = regex_to_postfix(regex)
            nfa = build_nfa_from_postfix(pf)
            img_path = f"AFN_{idx}.png"
            draw_nfa(nfa, img_path, title=f"AFN #{idx}")
            print(f"[{idx}] Expresión: {regex}")
            print(f"Postfix: {pf}")
            if args.w is not None:
                print(f"Cadena '{args.w}': {'sí' if nfa_accepts(nfa, args.w) else 'no'}")
            print(f"Imagen guardada en {img_path}\\n")
