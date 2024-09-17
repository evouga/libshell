#pragma once

namespace LibShell
{
    // Define the type of the Hessian projection
    enum class HessianProjectType
    {
        kNone,    // no projection
        kMaxZero, // project negative eigenvalues to zero
        kAbs      // project negative eigenvalues to their absolute values
    };
} // namespace LibShell