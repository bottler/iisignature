#ifndef CALCSUFFIXSIGNATURE_H
#define CALCSUFFIXSIGNATURE_H

#include <cmath>
#include <cstddef>
#include <vector>

// Number of signature components up to level m of a d-dimensional path
// whose word does NOT start with letter 0 (in 0-indexed storage terms,
// equivalently letter 1 in the user-facing 1-indexed alphabet). Equal to
// sum_{k=1..m} (d-1) * d^(k-1) = d^m - 1.
int calcSigTotalLengthSuffix(int d, int m) {
  int64_t p = static_cast<int64_t>(0.4 + std::pow(d, m));
  return (int)(p - 1);
}

namespace CalcSignature {

// SuffixSignature stores, for each level k = 1..m, only those d^k tensor
// components whose multi-index does not start with letter 0 -- i.e. the
// (d-1)*d^(k-1) "non-zero-led" suffix entries. Used by sigsuffix, the
// time-augmented feature-set described in Bourdon et al., arXiv:2601.10545.
//
// Algebraic trick. If we want the suffix part of (left o right) under Chen
// concatenation, the contribution at level L is
//     suffix(left)_L + suffix(left o right)_L cross-terms + suffix(right)_L
// Because dropping the first letter only sees the LEFT operand's first
// letter, the running accumulator (this class) can stay compact, while the
// per-segment operand (AdjointSuffixSignature) needs the full lower levels
// (1..m-1) and only the suffix at the top level m.

class AdjointSuffixSignature {
public:
  std::vector<std::vector<double>> m_data;

  void sigOfSegment(int d, int N, const double *segment) {
    if (m_data.size() != (size_t)N)
      m_data.resize(N);

    // Level 1: full d entries.
    m_data[0].assign(segment, segment + d);

    // Levels 2..N-1: full size d^k.
    for (int level = 2; level < N; ++level) {
      const auto &prev = m_data[level - 2];
      size_t L = prev.size() * (size_t)d;
      auto &cur = m_data[level - 1];
      if (cur.size() != L)
        cur.resize(L);
      const double inv = 1.0 / level;
      const double *__restrict__ pp = prev.data();
      double *__restrict__ cp = cur.data();
      const size_t P = prev.size();
      for (size_t i = 0; i < P; ++i) {
        const double scaled = pp[i] * inv;
        double *__restrict__ row = cp + i * (size_t)d;
        for (int a = 0; a < d; ++a)
          row[a] = segment[a] * scaled;
      }
    }

    // Level N: only the (d-1)*d^(N-1) suffix entries (first letter != 0).
    if (N >= 2) {
      const auto &prev = m_data[N - 2];
      size_t block = prev.size();
      size_t kept_len = (size_t)(d - 1) * block;
      auto &cur = m_data[N - 1];
      if (cur.size() != kept_len)
        cur.resize(kept_len);
      const double inv = 1.0 / N;
      const double *__restrict__ pp = prev.data();
      double *__restrict__ cp = cur.data();
      size_t idx = 0;
      for (int a = 1; a < d; ++a) {
        const double scaled = segment[a] * inv;
        for (size_t i = 0; i < block; ++i)
          cp[idx++] = pp[i] * scaled;
      }
    }
  }

  void sigOfNothing(int d, int N) {
    if (m_data.size() != (size_t)N)
      m_data.resize(N);
    m_data[0].assign((size_t)d, 0.0);
    size_t size = (size_t)d;
    for (int level = 2; level < N; ++level) {
      size *= (size_t)d;
      auto &cur = m_data[level - 1];
      cur.assign(size, 0.0);
    }
    if (N >= 2) {
      size_t kept_len = (size_t)(d - 1) * size;
      m_data[N - 1].assign(kept_len, 0.0);
    }
  }
};

class SuffixSignature {
public:
  std::vector<std::vector<double>> m_data;

  void sigOfSegment(int d, int m, const double *segment) {
    if (m_data.size() != (size_t)m)
      m_data.resize(m);

    // Level 1: (d-1) entries (drop letter 0 / 1-indexed letter 1).
    auto &lv1 = m_data[0];
    if (lv1.size() != (size_t)(d - 1))
      lv1.resize((size_t)(d - 1));
    for (int i = 1; i < d; ++i)
      lv1[i - 1] = segment[i];

    // Levels >= 2: stored compactly as (d-1)*d^(k-1). The lower-letter axis
    // expands by a factor of d each level, but the highest (first) letter is
    // still restricted to != 0, so the per-level layout is
    // [first_letter=1..d-1][lower-letters=0..d^(k-1)-1].
    for (int level = 2; level <= m; ++level) {
      const auto &last = m_data[level - 2];
      size_t L_last = last.size();
      // new size = d * old_size  (one new low-axis letter from 0..d-1)
      size_t new_size = L_last * (size_t)d;
      auto &cur = m_data[level - 1];
      if (cur.size() != new_size)
        cur.resize(new_size);
      const double inv = 1.0 / level;
      const double *__restrict__ lp = last.data();
      double *__restrict__ cp = cur.data();
      for (size_t i = 0; i < L_last; ++i) {
        const double scaled = lp[i] * inv;
        double *__restrict__ row = cp + i * (size_t)d;
        for (int a = 0; a < d; ++a)
          row[a] = segment[a] * scaled;
      }
    }
  }

  void sigOfNothing(int d, int m) {
    if (m_data.size() != (size_t)m)
      m_data.resize(m);
    m_data[0].assign((size_t)(d - 1), 0.0);
    size_t size = (size_t)(d - 1);
    for (int level = 2; level <= m; ++level) {
      size *= (size_t)d;
      m_data[level - 1].assign(size, 0.0);
    }
  }

  // In-place Chen concatenation: *this becomes the suffix signature of the
  // path made by joining the path of *this to the path of `other`. The
  // lower levels of `other` are stored full (d^k); only its top level is
  // compact, because only the top-level cross-product needs to project to
  // the suffix subspace -- below the top level, both operands' suffix parts
  // are read directly from compact storage.
  void concatenateWith(int d, int m, const AdjointSuffixSignature &other) {
    if (m_data.size() != (size_t)m)
      sigOfNothing(d, m);

    // Work top-down so that we read lower-level entries of *this before
    // overwriting them with cross-term contributions.
    for (int L = m; L >= 1; --L) {
      auto &destCompact = m_data[L - 1];
      const auto &srcOther = other.m_data[L - 1];

      // Add other's level L contribution. other's top level is already in
      // compact (d-1)*d^(L-1) layout; lower levels are full d^L and need
      // the first-letter=0 slab dropped before accumulating.
      if (L == m) {
        const size_t N = srcOther.size();
        const double *__restrict__ sp = srcOther.data();
        double *__restrict__ dp = destCompact.data();
        for (size_t i = 0; i < N; ++i)
          dp[i] += sp[i];
      } else {
        size_t block = 1;
        for (int i = 0; i < L - 1; ++i)
          block *= (size_t)d;
        const double *__restrict__ sp = srcOther.data();
        double *__restrict__ dp = destCompact.data();
        size_t pos = 0;
        for (int first = 1; first < d; ++first) {
          const double *__restrict__ slab = sp + (size_t)first * block;
          for (size_t off = 0; off < block; ++off)
            dp[pos++] += slab[off];
        }
      }

      // Cross terms: *this at level (L - otherlevel) tensor other at level
      // otherlevel, accumulated into destCompact (level L).
      for (int mylevel = L - 1; mylevel >= 1; --mylevel) {
        int otherlevel = L - mylevel;
        const auto &myVec = m_data[mylevel - 1];
        const auto &othFull = other.m_data[otherlevel - 1];
        const size_t Mm = myVec.size();
        const size_t Mo = othFull.size();
        const double *__restrict__ mp = myVec.data();
        const double *__restrict__ op = othFull.data();
        double *__restrict__ dp = destCompact.data();
        size_t write_pos = 0;
        for (size_t i = 0; i < Mm; ++i) {
          const double lv = mp[i];
          for (size_t j = 0; j < Mo; ++j)
            dp[write_pos++] += lv * op[j];
        }
      }
    }
  }

  template <typename Numeric> void writeOut(Numeric *dest) const {
    for (auto &a : m_data)
      for (auto &b : a)
        *(dest++) = (Numeric)b;
  }
};

// Suffix signature of a single path (lengthOfPath x d) with given strides
// (in bytes). The InputT template lets callers feed either float or double
// raw memory without an extra copy. The accumulator `out` is reused across
// calls for batched / parallel workloads.
template <typename InputT>
static void calcSignatureSuffix(SuffixSignature &out, const char *data,
                                std::ptrdiff_t point_stride,
                                std::ptrdiff_t dim_stride, int lengthOfPath,
                                int d, int level) {
  AdjointSuffixSignature adj;
  out.sigOfNothing(d, level);
  std::vector<double> displacement((size_t)d);
  for (int i = 1; i < lengthOfPath; ++i) {
    for (int j = 0; j < d; ++j)
      displacement[j] =
          (double)(*(const InputT *)(data + i * point_stride + j * dim_stride)) -
          (double)(*(const InputT *)(data + (i - 1) * point_stride +
                                     j * dim_stride));
    adj.sigOfSegment(d, level, displacement.data());
    out.concatenateWith(d, level, adj);
  }
}

} // namespace CalcSignature

#endif
