#pragma once

#include <chrono>
#include <stdexcept>

namespace OptSolver {
// Template definition allowing custom clock types, defaulting to
// high-resolution clock.
template <typename ClockType = std::chrono::high_resolution_clock> class Timer {
  // Time points to mark the start, pause, and end of the timer.
  std::chrono::time_point<ClockType> start_point;
  std::chrono::time_point<ClockType> pause_point;
  std::chrono::time_point<ClockType> end_point;

  // Flags to track the state of the timer.
  bool is_stopped;
  bool is_paused;

public:
  // Constructor initializes the timer and starts it immediately.
  Timer()
      : is_stopped(false), is_paused(false), start_point(ClockType::now()) {}

  // Default destructor.
  ~Timer() = default;

  // Starts or restarts the timer, resetting the stop and pause flags.
  void start() {
    start_point = ClockType::now();
    is_paused = false;
    is_stopped = false;
  }

  // Pauses the timer and marks the current time as the pause point.
  void pause() {
    if (!is_stopped &&
        !is_paused) { // Ensure the timer is running and not already paused.
      pause_point = ClockType::now();
      is_paused = true;
    }
  }

  // Resumes the timer from a paused state by adjusting the start time.
  void resume() {
    if (is_stopped) {
      throw std::runtime_error("Cannot resume a stopped timer.");
    }
    if (is_paused) {
      start_point += ClockType::now() - pause_point;
      is_paused = false;
    }
  }

  // Stops the timer and records the end time.
  void stop() {
    if (!is_stopped) { // Prevent stopping multiple times.
      end_point = ClockType::now();
      is_stopped = true;
    }
  }

  // Returns the elapsed time in the specified duration type.
  template <typename DurationType = std::chrono::milliseconds>
  typename DurationType::rep elapsed() const {
    if (is_stopped) {
      // Return the elapsed time between start and end if the timer is stopped.
      return std::chrono::duration_cast<DurationType>(end_point - start_point)
          .count();
    } else if (is_paused) {
      // Return the elapsed time until paused.
      return std::chrono::duration_cast<DurationType>(pause_point - start_point)
          .count();
    } else {
      // Return the currently elapsed time if the timer is running.
      return std::chrono::duration_cast<DurationType>(ClockType::now() -
                                                      start_point)
          .count();
    }
  }
};
} // namespace OptSolver