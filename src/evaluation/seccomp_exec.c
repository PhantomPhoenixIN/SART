/* Apply a conservative syscall filter, then exec one sandboxed benchmark task. */
#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <linux/sched.h>

#if !defined(__x86_64__)
#error "The SART seccomp launcher currently supports x86_64 only"
#endif

#define DENY_SYSCALL(number) \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (number), 0, 1), \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA))

static int install_filter(void) {
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
#ifdef __NR_socket
        DENY_SYSCALL(__NR_socket),
#endif
#ifdef __NR_socketpair
        DENY_SYSCALL(__NR_socketpair),
#endif
#ifdef __NR_connect
        DENY_SYSCALL(__NR_connect),
#endif
#ifdef __NR_bind
        DENY_SYSCALL(__NR_bind),
#endif
#ifdef __NR_listen
        DENY_SYSCALL(__NR_listen),
#endif
#ifdef __NR_accept
        DENY_SYSCALL(__NR_accept),
#endif
#ifdef __NR_accept4
        DENY_SYSCALL(__NR_accept4),
#endif
#ifdef __NR_sendto
        DENY_SYSCALL(__NR_sendto),
#endif
#ifdef __NR_recvfrom
        DENY_SYSCALL(__NR_recvfrom),
#endif
#ifdef __NR_fork
        DENY_SYSCALL(__NR_fork),
#endif
#ifdef __NR_vfork
        DENY_SYSCALL(__NR_vfork),
#endif
#ifdef __NR_clone3
        DENY_SYSCALL(__NR_clone3),
#endif
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 offsetof(struct seccomp_data, args[0])),
        BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, CLONE_THREAD, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
#ifdef __NR_ptrace
        DENY_SYSCALL(__NR_ptrace),
#endif
#ifdef __NR_process_vm_readv
        DENY_SYSCALL(__NR_process_vm_readv),
#endif
#ifdef __NR_process_vm_writev
        DENY_SYSCALL(__NR_process_vm_writev),
#endif
#ifdef __NR_kcmp
        DENY_SYSCALL(__NR_kcmp),
#endif
#ifdef __NR_pidfd_getfd
        DENY_SYSCALL(__NR_pidfd_getfd),
#endif
#ifdef __NR_mount
        DENY_SYSCALL(__NR_mount),
#endif
#ifdef __NR_umount2
        DENY_SYSCALL(__NR_umount2),
#endif
#ifdef __NR_pivot_root
        DENY_SYSCALL(__NR_pivot_root),
#endif
#ifdef __NR_setns
        DENY_SYSCALL(__NR_setns),
#endif
#ifdef __NR_unshare
        DENY_SYSCALL(__NR_unshare),
#endif
#ifdef __NR_bpf
        DENY_SYSCALL(__NR_bpf),
#endif
#ifdef __NR_perf_event_open
        DENY_SYSCALL(__NR_perf_event_open),
#endif
#ifdef __NR_userfaultfd
        DENY_SYSCALL(__NR_userfaultfd),
#endif
#ifdef __NR_keyctl
        DENY_SYSCALL(__NR_keyctl),
#endif
#ifdef __NR_open_by_handle_at
        DENY_SYSCALL(__NR_open_by_handle_at),
#endif
#ifdef __NR_io_uring_setup
        DENY_SYSCALL(__NR_io_uring_setup),
#endif
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
        .filter = filter,
    };

    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        perror("prctl(PR_SET_NO_NEW_PRIVS)");
        return -1;
    }
    if (syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &program) != 0) {
        perror("seccomp(SECCOMP_SET_MODE_FILTER)");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s PROGRAM [ARG ...]\n", argv[0]);
        return 64;
    }
    if (clearenv() != 0 || setenv("PATH", "/usr/bin:/bin", 1) != 0 ||
        setenv("HOME", "/tmp", 1) != 0 || setenv("TMPDIR", "/tmp", 1) != 0 ||
        setenv("LANG", "C.UTF-8", 1) != 0) {
        perror("sanitize environment");
        return 70;
    }
    if (install_filter() != 0) {
        return 70;
    }
    execvp(argv[1], &argv[1]);
    perror("execvp");
    return errno == ENOENT ? 127 : 126;
}
