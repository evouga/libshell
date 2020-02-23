#include "findiff.h"
#include <iostream>
#include <cmath>
#include <algorithm>

void FiniteDifferenceLog::addEntry(double epsilon, double derivative, double centerdifference)
{
    entries_[epsilon].push_back({ derivative, centerdifference });
}

void FiniteDifferenceLog::clear()
{
    entries_.clear();
}

void FiniteDifferenceLog::printStats()
{
    std::cout << "Epsilon: Error" << std::endl;
    for (auto &it : entries_)
    {
        std::cout << it.first << ": ";
        double maxerror = 0;
        for (auto &entry : it.second)
        {
            double absolute = std::fabs(entry.deriv - entry.difference);
            double relative;
            if (entry.deriv == 0.0 && entry.difference == 0.0)
                relative = 0.0;
            else
                relative = std::fabs( (entry.difference - entry.deriv) / entry.difference );
            double localerror = std::min(relative, absolute);
            maxerror = std::max(maxerror, localerror);
        }
        std::cout << maxerror << std::endl;
    }
}