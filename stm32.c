// throttle_tuning.c
// Ported from the uploaded Python script.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>

// ============================================================
// TUNING PARAMETERS — adjust these
// ============================================================

#define MAX_THROTTLE_PCT 0.15   // 15%
#define RAMP_TIME_SEC    2.0    // seconds to ramp from 0 to max

// ============================================================
// CONFIG — don't touch these unless hardware changes
// ============================================================

#define PORT "/dev/ttyAMA0"
#define BAUD B115200

#define ESC_LEFT_CH   14
#define ESC_RIGHT_CH  15
#define ESC_NEUTRAL   90
#define ESC_MIN       30
#define ESC_MAX       150

#define DEADBAND_LOW   1350
#define DEADBAND_HIGH  1650

// derived from tuning params
#define MAX_POWER     (MAX_THROTTLE_PCT * 100.0)
#define THROTTLE_RAMP (MAX_POWER / (RAMP_TIME_SEC * 50.0))

// ============================================================
// HARDWARE ABSTRACTION LAYER
// Replace these with your real servo/PCA9685 functions
// ============================================================

static int pca9685_init(void) {
    // TODO: initialize your PCA9685 or servo driver here
    // return 0 on success, nonzero on failure
    return 0;
}

static void pca9685_set_servo_angle(int channel, int angle) {
    // TODO: send servo angle to hardware here
    // For now, stub only
    (void)channel;
    (void)angle;
}

// ============================================================
// UTILS
// ============================================================

static double clamp(double val, double lo, double hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

static double map_range(double x, double in_min, double in_max,
                        double out_min, double out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

static double apply_deadband(int val, int low, int high) {
    if (val >= low && val <= high) {
        return 0.0;
    } else if (val > high) {
        return map_range(val, high, 2000, 0, 100);
    } else {
        return map_range(val, 1000, low, -100, 0);
    }
}

static void set_esc(int ch, double value) {
    int angle = (int)clamp(value, ESC_MIN, ESC_MAX);
    pca9685_set_servo_angle(ch, angle);
}

static void neutral_escs(void) {
    pca9685_set_servo_angle(ESC_LEFT_CH, ESC_NEUTRAL);
    pca9685_set_servo_angle(ESC_RIGHT_CH, ESC_NEUTRAL);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// ============================================================
// SERIAL
// ============================================================

static int open_serial(const char *port) {
    int fd = open(port, O_RDONLY | O_NOCTTY);
    if (fd < 0) {
        perror("open serial");
        return -1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));

    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        close(fd);
        return -1;
    }

    cfsetispeed(&tty, BAUD);
    cfsetospeed(&tty, BAUD);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_iflag = 0;
    tty.c_oflag = 0;
    tty.c_lflag = 0;

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1; // 0.1s read timeout

    tcflush(fd, TCIFLUSH);

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        close(fd);
        return -1;
    }

    return fd;
}

// ============================================================
// BUFFER HELPERS
// ============================================================

#define BUF_CAP 512

typedef struct {
    uint8_t data[BUF_CAP];
    size_t len;
} ByteBuffer;

static void buf_append(ByteBuffer *buf, const uint8_t *src, size_t n) {
    if (n > BUF_CAP - buf->len) {
        size_t overflow = n - (BUF_CAP - buf->len);
        if (overflow >= buf->len) {
            buf->len = 0;
        } else {
            memmove(buf->data, buf->data + overflow, buf->len - overflow);
            buf->len -= overflow;
        }
    }
    memcpy(buf->data + buf->len, src, n);
    buf->len += n;
}

static int buf_find_header(const ByteBuffer *buf) {
    for (size_t i = 0; i + 1 < buf->len; i++) {
        if (buf->data[i] == 0x20 && buf->data[i + 1] == 0x40) {
            return (int)i;
        }
    }
    return -1;
}

static void buf_discard_front(ByteBuffer *buf, size_t n) {
    if (n >= buf->len) {
        buf->len = 0;
        return;
    }
    memmove(buf->data, buf->data + n, buf->len - n);
    buf->len -= n;
}

// ============================================================
// MAIN
// ============================================================

int main(void) {
    printf("==================================================\n");
    printf("THROTTLE TUNING SCRIPT\n");
    printf("==================================================\n");
    printf("  Max throttle : %.0f%%\n", MAX_THROTTLE_PCT * 100.0);
    printf("  Ramp time    : %.1f seconds to reach max\n", RAMP_TIME_SEC);
    printf("  Ramp rate    : %.3f per loop\n", THROTTLE_RAMP);
    printf("==================================================\n");
    printf("CH7 kill switch to enable | Ctrl+C to stop\n\n");

    if (pca9685_init() != 0) {
        fprintf(stderr, "Failed to initialize servo driver\n");
        return 1;
    }

    int fd = open_serial(PORT);
    if (fd < 0) {
        return 1;
    }

    ByteBuffer buf = {0};
    bool valid_signal = false;
    double current_throttle = 0.0;
    double last_print = 0.0;

    printf("Arming ESCs...\n");
    neutral_escs();
    sleep(2);
    printf("Ready.\n\n");

    while (1) {
        uint8_t temp[32];
        ssize_t n = read(fd, temp, sizeof(temp));
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            perror("read");
            break;
        }
        if (n == 0) {
            continue;
        }

        buf_append(&buf, temp, (size_t)n);

        while (buf.len >= 32) {
            int idx = buf_find_header(&buf);
            if (idx < 0) {
                if (buf.len > 1) {
                    uint8_t last = buf.data[buf.len - 1];
                    buf.len = 1;
                    buf.data[0] = last;
                }
                break;
            }

            if (idx > 0) {
                buf_discard_front(&buf, (size_t)idx);
            }

            if (buf.len < 32) {
                break;
            }

            uint8_t frame[32];
            memcpy(frame, buf.data, 32);

            uint16_t chk = 0xFFFF;
            for (int i = 0; i < 30; i++) {
                chk -= frame[i];
            }

            uint16_t frame_chk = (uint16_t)(frame[30] | (frame[31] << 8));
            if (chk != frame_chk) {
                buf_discard_front(&buf, 1);
                continue;
            }

            int channels[14];
            for (int i = 0; i < 14; i++) {
                channels[i] = (int)(frame[2 + i * 2] | (frame[3 + i * 2] << 8));
            }

            buf_discard_front(&buf, 32);

            bool channels_ok = true;
            for (int i = 0; i < 10; i++) {
                if (channels[i] < 900 || channels[i] > 2100) {
                    channels_ok = false;
                    break;
                }
            }
            if (!channels_ok) {
                continue;
            }

            if (!valid_signal) {
                printf("RC signal acquired!\n");
                valid_signal = true;
            }

            int ch2_throttle = channels[1];
            int ch7_kill     = channels[6];
            int ch8_reverse  = channels[7];

            bool is_enabled = valid_signal && (ch7_kill > 1500);
            const char *status = "DISABLED";

            if (!is_enabled) {
                neutral_escs();
                current_throttle = 0.0;
                status = "DISABLED";
            } else {
                double raw = apply_deadband(ch2_throttle, DEADBAND_LOW, DEADBAND_HIGH);
                raw = clamp(raw, -MAX_POWER, MAX_POWER);

                double target = (ch8_reverse > 1500) ? -raw : raw;

                double diff = target - current_throttle;
                if (fabs(diff) < THROTTLE_RAMP) {
                    current_throttle = target;
                } else if (diff > 0) {
                    current_throttle += THROTTLE_RAMP;
                } else {
                    current_throttle -= THROTTLE_RAMP;
                }

                if (fabs(current_throttle) < 0.5) {
                    neutral_escs();
                    current_throttle = 0.0;
                    status = "NEUTRAL";
                } else {
                    double esc_val = map_range(current_throttle,
                                               -MAX_POWER, MAX_POWER,
                                               ESC_MIN, ESC_MAX);
                    set_esc(ESC_LEFT_CH, esc_val);
                    set_esc(ESC_RIGHT_CH, esc_val);
                    status = (current_throttle > 0) ? "ACTIVE FWD" : "ACTIVE REV";
                }
            }

            double t = now_sec();
            if (t - last_print > 0.2) {
                double esc_val = map_range(current_throttle,
                                           -MAX_POWER, MAX_POWER,
                                           ESC_MIN, ESC_MAX);

                printf("[%s] raw_ch:%d | throttle:%+.1f/%.0f | ESC:%.1f | ramp:%.3f/loop\n",
                       status,
                       ch2_throttle,
                       current_throttle,
                       MAX_POWER,
                       esc_val,
                       THROTTLE_RAMP);
                fflush(stdout);
                last_print = t;
            }
        }
    }

    printf("\nStopping...\n");
    neutral_escs();
    close(fd);
    printf("Done.\n");
    return 0;
}