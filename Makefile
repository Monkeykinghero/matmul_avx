CC       = gcc
CFLAGS   = -O2 -mavx -mfma -march=native -Wall -Wextra
LDFLAGS  =
TARGET   = matmul_avx.exe
SRC      = matmul_avx.c

# 允许覆盖 N 和 TILE: make N=1024 TILE=128
N    ?= 512
TILE ?= 64

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -DN=$(N) -DTILE=$(TILE) -o $(TARGET) $(SRC) $(LDFLAGS)

.PHONY: clean run

run: $(TARGET)
	./$(TARGET)

clean:
	del /Q $(TARGET) 2>nul || exit 0
