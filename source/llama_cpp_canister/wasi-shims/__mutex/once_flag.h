// WASI shim: <__mutex/once_flag.h> — single-threaded no-op
#ifndef _LIBCPP___MUTEX_ONCE_FLAG_H
#define _LIBCPP___MUTEX_ONCE_FLAG_H

namespace std {

struct once_flag {
    bool called_ = false;
};

template <class Callable, class... Args>
void call_once(once_flag& flag, Callable&& f, Args&&... args) {
    if (!flag.called_) {
        flag.called_ = true;
        f(static_cast<Args&&>(args)...);
    }
}

} // namespace std

#endif // _LIBCPP___MUTEX_ONCE_FLAG_H
