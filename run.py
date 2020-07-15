#!/usr/bin/env pypy3
import argparse
import numpy as np
import os
import sys

# We want to allow OpenMP to cancel a thread if no more work needs to be done.
# It is very important that this environment variable is set BEFORE importing
# the `nlnum` module.
os.environ['OMP_CANCELLATION'] = 'true'

from collections import namedtuple
from functools import lru_cache, partial
from itertools import combinations, permutations, product
from nlnum import PartitionsIn, nlcoef as nlc, lrcoef as lrc


Triple = namedtuple('Triple', ['mu', 'nu', 'lam'])
Result = namedtuple('Result', ['triple', 'satisfies', 'failed_inequalities'])



@lru_cache(maxsize=None)
def _lrcoef(o, i1, i2, *args, **kwargs):
  return lrc(o, i1, i2, *args, **kwargs)

def lrcoef(outer, inner1, inner2, *args, **kwargs):
  return _lrcoef(tuple(outer), tuple(inner1), tuple(inner2), *args, **kwargs)

@lru_cache(maxsize=None)
def _nlcoef(l, m, n, *args, **kwargs):
  return nlc(l, m, n, *args, **kwargs)

def nlcoef(lam, mu, nu, *args, **kwargs):
  return _nlcoef(tuple(lam), tuple(mu), tuple(nu), *args, **kwargs)


@lru_cache(maxsize=None)
def disjoints(n, nsets):
  '''
  Returns all disjoint sets of {1,...,n} into `nsets` sets.
  This is not memory efficient.
  '''
  ds = []
  for bits in product(range(nsets+1), repeat=n):
    S = [list() for _ in range(nsets+1)]
    for i, bit in enumerate(bits):
      S[bit].append(i+1)
    ds.append(tuple(S[1:]))
  return ds


def at(partition, idx):
  if idx > len(partition): return 0
  if idx >= 1: return partition[idx-1]
  raise IndexError(f'Negative index {idx} is out of bounds.')


def flagger(n, k, inequalities=[]):
  for km in range(k+1):
    for kn in range(km+1):
      # Only look at kl that satisfies the triangle inequality.
      for kl in range(abs(km-kn), 1+min(kn, kn+km)):
        if (km + kn + kl) % 2 == 1: continue

        pp = product(*[PartitionsIn([kk]*n, kk) for kk in (km, kn, kl)])
        for mu, nu, lam in pp:
          if not (mu >= nu >= lam): continue

          # If NL > 0 and a perm fails the ineq OR NL = 0 and all perms succeed,
          # then yield the triple.
          satisfies = True
          pos = bool(nlcoef(mu, nu, lam, check_positivity=True))
          for (muu, nuu, lamm) in permutations((mu, nu, lam)):
            sat = all(ineq(muu, nuu, lamm) for ineq in inequalities)
            satisfies = satisfies and sat
            if pos and not sat:
              failed_inequalities = [
                (i, (muu, nuu, lamm))
                for i, ineq in enumerate(inequalities)
                if not ineq(muu, nuu, lamm)
              ]
              yield Result(
                Triple(mu, nu, lam),
                satisfies=False,
                failed_inequalities=failed_inequalities,
              )
              break
          else:
            if not pos and satisfies:
              yield Result(
                Triple(mu, nu, lam),
                satisfies=True,
                failed_inequalities=None,
              )

  yield from ()


def theorem512(n, mu, nu, lam, verbose=False):
  for k in range(1, n):
    for i in range(k, n):
      for j in range(i+1, n+1):
        for l in range(j, n+1):
          m, M = sorted((i-k, l-j))
          if not all(
            0 <= -at(mu, i) + at(mu, j) + at(nu, k) - at(nu, l) + at(lam, m-p+1) + at(lam, M+p+2)
            for p in range(m+1)
          ):
            if verbose: print([
              (k, i, j, l, p, m, M)
              for p in range(m+1)
              if not (0 <= -at(mu, i) + at(mu, j) + at(nu, k) - at(nu, l) + at(lam, m-p+1) + at(lam, M+p+2))
            ])
            return False
  return True


def dsums(n, mu, nu, lam, verbose=False):
  for S, T in disjoints(n, nsets=2):
    if not (0 <= sum(at(mu, i) - at(nu, i) for i in S) + sum(at(nu, i) - at(mu, i) for i in T) + sum(at(lam, i) for i in range(1, len(S)+len(T)+1))):
      if verbose:
        print((S, T))
      return False
  return True


def tau(X):
  d = len(X)
  ss = np.array(sorted(X))
  rr = 1+np.arange(d)
  tt = list(ss-rr)[::-1]
  return tt


@lru_cache(maxsize=None)
def horns(n):
  '''
  Generate the I, J, K for horns inequality.
  We now sacrifice memory.
  '''
  return [
    (I, J, K)
    for d in range(n)
    for I, J, K in product(combinations(range(1, n+1), r=d), repeat=3)
    if lrcoef(tau(K), tau(I), tau(J)) > 0
  ]


def horn(n, mu, nu, lam, verbose=False):
  for I, J, K in horns(n):
    if not (sum(at(lam, k) for k in K) <= sum(at(mu, i) for i in I) + sum(at(nu, j) for j in J)):
      if verbose: print(
        n, mu, nu, lam, I, J, K,
      )
      return False
  return True


# Every triple with positive NL-number should satisfy these inequalities.
get_inequalities = lambda n: [
    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(lam, 1) - at(lam, 3) + at(nu, 1) + at(nu, 3),
    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(mu, 3) + at(lam, 1) - at(lam, 2) + at(lam, 3) + at(nu, 1) + at(nu, 2) - at(nu, 3),
    # lambda mu, nu, lam: 0 <= -at(mu, 2) + at(lam, 1) + at(nu, 2),

    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(mu, 3)            + at(lam, 1) - at(lam, 2) + at(lam, 3)              + at(nu, 1) + at(nu, 2) - at(nu, 3),
    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(mu, 3)            + at(lam, 1) - at(lam, 2) + at(lam, 4)              + at(nu, 1) + at(nu, 2) - at(nu, 4),
    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(mu, 4)            + at(lam, 1) - at(lam, 2) + at(lam, 3)              + at(nu, 1) + at(nu, 2) - at(nu, 4),
    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(mu, 3)            + at(lam, 1) - at(lam, 3) + at(lam, 4)              + at(nu, 1) + at(nu, 3) - at(nu, 4),
    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(mu, 4)            + at(lam, 1) - at(lam, 3) + at(lam, 4)              + at(nu, 1) + at(nu, 2) - at(nu, 4),
    # lambda mu, nu, lam: 0 <= -at(mu, 2) + at(mu, 3) + at(mu, 4)            + at(lam, 1) - at(lam, 3) + at(lam, 4)              + at(nu, 1) + at(nu, 2) - at(nu, 4),

    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(mu, 3) + at(mu,4) + at(lam, 1) - at(lam, 2) + at(lam, 3) + at(lam, 4) + at(nu, 1) + at(nu, 2) - at(nu, 3) - at(nu, 4),
    # lambda mu, nu, lam: 0 <= -at(mu, 1) + at(mu, 2) + at(mu, 3) + at(mu,4) + at(lam, 1) - at(lam, 2) - at(lam, 3) + at(lam, 4) + at(nu, 1) + at(nu, 2) + at(nu, 3) - at(nu, 4),
    # lambda mu, nu, lam: 0 <= -at(mu, 1) - at(mu, 2) + at(mu, 3) + at(mu,4) + at(lam, 1) + at(lam, 2) - at(lam, 3) + at(lam, 4) + at(nu, 1) + at(nu, 2) + at(nu, 3) - at(nu, 4),

    partial(dsums, n),
    partial(theorem512, n),
    partial(horn, n),
  ]


def main(args):
  n, k = args.n, args.k

  inequalities = get_inequalities(n)

  n_results = 0
  for result in flagger(n, k, inequalities=inequalities):
    print(result)
    n_results += 1

  print(f'Number flagged: {n_results}')
  # The `failed_inequalities` field is a list of tuples (i, (m,n,l)) where `i`
  # is the index of the inequality that failed and `m,n,l` is the permutation of
  # the partitions that caused the inequality to fail


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Check inequalities related to NL numbers.')
  parser.add_argument('-n', required=True, type=int, help='maximum length of a partition')
  parser.add_argument('-k', required=True, type=int, help='maximum size of a partition')
  args = parser.parse_args()
  sys.exit(main(args))
