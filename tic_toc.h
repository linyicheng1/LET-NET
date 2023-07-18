#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
	    end = std::chrono::system_clock::now();
		auto count = (double )std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	    return count;
    }
	double fps()
	{
		end = std::chrono::system_clock::now();
		auto count = (double )std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		return (double)1./count * 1000000;
	}
  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

