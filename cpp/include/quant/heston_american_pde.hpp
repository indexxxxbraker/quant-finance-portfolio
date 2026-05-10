// heston_american_pde.hpp
//
// Heston model: American put pricing via 2D PDE with operator-splitting projection.
//
// Extends the European 2D PDE pricer of Block 5 with a projection
// step at each time level to enforce the early-exercise constraint
// V(t, S, v) >= max(K - S, 0). The numerical scheme is operator
// splitting: one Douglas ADI step (solving the European Heston PDE)
// followed by a max-projection in the (X = log S, tau = T - t,
// W = e^{r tau} U) coordinates.
//
// See theory/phase4/block6_heston_calibration_exotics.tex Section 4.4
// for the rationale (PSOR-on-PDE vs Longstaff-Schwartz MC) and the
// derivation of the projection in W coordinates.
//
// The kernel functions (Thomas batched solver, operator coefficients,
// operator application, implicit solves, Douglas step) are reproduced
// from heston_pde.cpp for self-containment.

#pragma once

#include "heston_fourier.hpp"   // HestonParams

#include <cstddef>

namespace quant::heston {

// Price an American put under Heston via 2D PDE with operator-splitting
// projection. The projection enforces V >= max(K - S, 0) at every
// time step.
//
// Parameters: same as heston_call_pde from Block 5, but the payoff
// is the put intrinsic max(K - S, 0) instead of max(S - K, 0).
//
// Returns: American put price.
//
// Notes:
//   - American put price has no Fourier ground truth; sanity bounds
//     are American >= European (early exercise premium >= 0) and
//     American = max(K - S0, 0) for deep ITM.
//   - Operator-splitting accuracy: the EEP estimate increases with
//     N_tau refinement (the projection is applied N_tau times). For
//     production accuracy, consider PSOR-within-sweep instead.
double heston_american_put_pde(double S0,
                                  double K,
                                  double T,
                                  double r,
                                  const HestonParams& p,
                                  std::size_t N_X = 200,
                                  std::size_t N_v = 100,
                                  std::size_t N_tau = 100,
                                  double theta_imp = 0.5,
                                  double X_factor = 4.0,
                                  double v_max_factor = 5.0);

}  // namespace quant::heston
