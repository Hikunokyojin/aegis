#include <cstring>
#include <cstdio>

void copy_input(const char* user_input) {
    char buffer[64];
    strcpy(buffer, user_input);
    printf("Copied: %s\n", buffer);
}

int main(int argc, char** argv) {
    if (argc > 1) {
        copy_input(argv[1]);
    }
    return 0;
}
