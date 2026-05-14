CC       = D:\MinGW\bin\gcc
CFLAGS   = -O2 -mavx -mfma -march=native -Wall -Wextra
LDFLAGS  =
TARGET   = matmul_avx.exe
SRC      = matmul_avx.c

# 允许覆盖 MATN 和 MAT_TILE: make MATN=1024 MAT_TILE=128
MATN    ?= 512
MAT_TILE ?= 64

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -DMATN=$(MATN) -DMAT_TILE=$(MAT_TILE) -o $(TARGET) $(SRC) $(LDFLAGS)

.PHONY: clean run

run: $(TARGET)
	./$(TARGET)

clean:
	del /Q $(TARGET) 2>nul || exit 0
