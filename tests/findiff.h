#ifndef FINDIFF_H
#define FINDIFF_H

#include <map>
#include <vector>

struct LogEntry
{
    double deriv;
    double difference;
};

class FiniteDifferenceLog
{
public:
    void addEntry(double epsilon, double derivative, double centerdifference);
    void clear();
    void printStats();

private:
    std::map<double, std::vector<LogEntry> > entries_;
};


#endif