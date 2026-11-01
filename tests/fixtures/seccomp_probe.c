#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

int main(void) {
    if (getenv("HF_TOKEN") != NULL || getenv("AWS_SECRET_ACCESS_KEY") != NULL ||
        getenv("HOME") == NULL || strcmp(getenv("HOME"), "/tmp") != 0 ||
        prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1 ||
        prctl(PR_GET_SECCOMP, 0, 0, 0, 0) != 2) {
        fputs("environment or privilege filter was not applied\n", stderr);
        return 3;
    }
    int network_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (network_socket != -1 || errno != EPERM) {
        fprintf(stderr, "network syscall was not rejected (fd=%d errno=%d)\n",
                network_socket, errno);
        return 1;
    }
    pid_t child = fork();
    if (child != -1 || errno != EPERM) {
        fprintf(stderr, "fork was not rejected (pid=%d errno=%d)\n", (int)child,
                errno);
        return 2;
    }
    puts("SECCOMP_PROBE_OK");
    return 0;
}
