#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include <vector>

const int MAIN_LEADER_ID = 0;
const int INTERNAL_LEADER_ID = 0;

static inline bool isMainLeader(int processId) { return processId == MAIN_LEADER_ID; }

enum class Algorithm { ColumnA, InnerABC };

static inline std::ostream& operator<<(std::ostream& os, Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::ColumnA:
            os << "ColumnA";
            break;
        case Algorithm::InnerABC:
            os << "InnerAvc";
            break;
    }
    return os;
}

#endif /* __COMMON_H__ */