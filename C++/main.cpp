#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <net.h>

struct Options
{
    std::string model_param = "model.ncnn.param";
    std::string model_bin = "model.ncnn.bin";
    std::string serial_port = "/dev/ttyUSB0";
    int camera_index = 0;
    int input_size = 320;
    int baudrate = 115200;
    float conf_thres = 0.25f;
    float nms_thres = 0.45f;
    bool show_window = true;
};

struct LetterboxInfo
{
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int resized_w = 0;
    int resized_h = 0;
};

struct Detection
{
    cv::Rect2f box;
    float score = 0.0f;
    int class_id = -1;
};

static int baud_to_flag(int baudrate)
{
    switch (baudrate)
    {
    case 9600:
        return B9600;
    case 19200:
        return B19200;
    case 38400:
        return B38400;
    case 57600:
        return B57600;
    case 115200:
        return B115200;
    case 230400:
        return B230400;
    case 460800:
        return B460800;
    case 921600:
        return B921600;
    default:
        return B115200;
    }
}

static Options parse_args(int argc, char **argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto need_value = [&](const std::string &name) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--model")
        {
            opt.model_param = need_value(arg);
        }
        else if (arg == "--bin")
        {
            opt.model_bin = need_value(arg);
        }
        else if (arg == "--serial")
        {
            opt.serial_port = need_value(arg);
        }
        else if (arg == "--camera")
        {
            opt.camera_index = std::stoi(need_value(arg));
        }
        else if (arg == "--size")
        {
            opt.input_size = std::stoi(need_value(arg));
        }
        else if (arg == "--baud")
        {
            opt.baudrate = std::stoi(need_value(arg));
        }
        else if (arg == "--conf")
        {
            opt.conf_thres = std::stof(need_value(arg));
        }
        else if (arg == "--nms")
        {
            opt.nms_thres = std::stof(need_value(arg));
        }
        else if (arg == "--no-show")
        {
            opt.show_window = false;
        }
    }
    return opt;
}

static int open_serial(const std::string &port_name, int baudrate)
{
    int fd = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0)
    {
        std::cerr << "Failed to open serial port: " << port_name << std::endl;
        return -1;
    }

    struct termios tty{};
    if (tcgetattr(fd, &tty) != 0)
    {
        std::cerr << "tcgetattr failed." << std::endl;
        close(fd);
        return -1;
    }

    cfmakeraw(&tty);
    speed_t speed = baud_to_flag(baudrate);
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(fd, TCSANOW, &tty) != 0)
    {
        std::cerr << "tcsetattr failed." << std::endl;
        close(fd);
        return -1;
    }

    return fd;
}

static void send_delta(int fd, int16_t delta_x)
{
    if (fd < 0)
    {
        return;
    }

    uint8_t packet[4];
    packet[0] = 0xAA;
    packet[1] = static_cast<uint8_t>((delta_x >> 8) & 0xFF);
    packet[2] = static_cast<uint8_t>(delta_x & 0xFF);
    packet[3] = 0xBB;
    (void)write(fd, packet, sizeof(packet));
}

static LetterboxInfo letterbox(const cv::Mat &src, cv::Mat &dst, int size)
{
    LetterboxInfo info;
    const int src_w = src.cols;
    const int src_h = src.rows;

    info.scale = std::min(static_cast<float>(size) / src_w, static_cast<float>(size) / src_h);
    info.resized_w = static_cast<int>(std::round(src_w * info.scale));
    info.resized_h = static_cast<int>(std::round(src_h * info.scale));
    info.pad_x = (size - info.resized_w) / 2;
    info.pad_y = (size - info.resized_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(info.resized_w, info.resized_h));
    dst = cv::Mat(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(dst(cv::Rect(info.pad_x, info.pad_y, info.resized_w, info.resized_h)));
    return info;
}

static inline float get_out_value(const ncnn::Mat &out, int row, int col)
{
    if (out.dims != 2)
    {
        return 0.0f;
    }

    if (out.h <= out.w)
    {
        return out.row(row)[col];
    }

    return out.row(col)[row];
}

static std::vector<Detection> decode_detections(const ncnn::Mat &out,
                                                int img_w,
                                                int img_h,
                                                const LetterboxInfo &lb,
                                                float conf_thres,
                                                float nms_thres)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    const int num_candidates = (out.h <= out.w) ? out.w : out.h;
    const bool normal_layout = (out.h <= out.w);

    for (int i = 0; i < num_candidates; ++i)
    {
        float v0 = normal_layout ? out.row(0)[i] : out.row(i)[0];
        float v1 = normal_layout ? out.row(1)[i] : out.row(i)[1];
        float v2 = normal_layout ? out.row(2)[i] : out.row(i)[2];
        float v3 = normal_layout ? out.row(3)[i] : out.row(i)[3];
        float s0 = normal_layout ? out.row(4)[i] : out.row(i)[4];
        float s1 = normal_layout ? out.row(5)[i] : out.row(i)[5];
        float s2 = normal_layout ? out.row(6)[i] : out.row(i)[6];

        int class_id = 0;
        float score = s0;
        if (s1 > score)
        {
            score = s1;
            class_id = 1;
        }
        if (s2 > score)
        {
            score = s2;
            class_id = 2;
        }

        if (score < conf_thres)
        {
            continue;
        }

        float x1, y1, x2, y2;
        if (v2 > v0 && v3 > v1)
        {
            x1 = v0;
            y1 = v1;
            x2 = v2;
            y2 = v3;
        }
        else
        {
            const float cx = v0;
            const float cy = v1;
            const float w = v2;
            const float h = v3;
            x1 = cx - w * 0.5f;
            y1 = cy - h * 0.5f;
            x2 = cx + w * 0.5f;
            y2 = cy + h * 0.5f;
        }

        x1 = (x1 - lb.pad_x) / lb.scale;
        y1 = (y1 - lb.pad_y) / lb.scale;
        x2 = (x2 - lb.pad_x) / lb.scale;
        y2 = (y2 - lb.pad_y) / lb.scale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_w - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_h - 1)));

        const float bw = x2 - x1;
        const float bh = y2 - y1;
        if (bw <= 1.0f || bh <= 1.0f)
        {
            continue;
        }

        boxes.emplace_back(cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                    cv::Size(static_cast<int>(bw), static_cast<int>(bh))));
        scores.emplace_back(score);
        class_ids.emplace_back(class_id);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_thres, nms_thres, keep);

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (int idx : keep)
    {
        Detection det;
        det.box = cv::Rect2f(static_cast<float>(boxes[idx].x),
                             static_cast<float>(boxes[idx].y),
                             static_cast<float>(boxes[idx].width),
                             static_cast<float>(boxes[idx].height));
        det.score = scores[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }

    std::sort(detections.begin(), detections.end(), [](const Detection &a, const Detection &b)
              { return a.score > b.score; });
    return detections;
}

static std::string class_name(int class_id)
{
    switch (class_id)
    {
    case 0:
        return "buff_block";
    case 1:
        return "debuff_block";
    case 2:
        return "other_block";
    default:
        return "unknown";
    }
}

static void draw_detections(cv::Mat &image, const std::vector<Detection> &detections)
{
    for (const auto &det : detections)
    {
        cv::Scalar color(0, 255, 0);
        if (det.class_id == 1)
            color = cv::Scalar(0, 255, 255);
        if (det.class_id == 2)
            color = cv::Scalar(255, 0, 255);

        cv::rectangle(image, det.box, color, 2);
        std::string label = class_name(det.class_id) + " " + cv::format("%.2f", det.score);
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int x = std::max(0, static_cast<int>(det.box.x));
        int y = std::max(0, static_cast<int>(det.box.y) - label_size.height - 4);
        cv::rectangle(image, cv::Rect(x, y, label_size.width + 4, label_size.height + 4), color, cv::FILLED);
        cv::putText(image, label, cv::Point(x + 2, y + label_size.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

int main(int argc, char **argv)
{
    try
    {
        Options opt = parse_args(argc, argv);

        ncnn::Net net;
        net.load_param(opt.model_param.c_str());
        net.load_model(opt.model_bin.c_str());
        net.opt.num_threads = std::max(1, cv::getNumberOfCPUs() - 1);

        int serial_fd = open_serial(opt.serial_port, opt.baudrate);
        cv::VideoCapture cap(opt.camera_index, cv::CAP_V4L2);
        if (!cap.isOpened())
        {
            cap.open(opt.camera_index);
        }
        if (!cap.isOpened())
        {
            std::cerr << "Failed to open camera." << std::endl;
            return 1;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
        cap.set(cv::CAP_PROP_FPS, 30);

        cv::Mat frame;
        cv::Mat letterboxed;
        cv::Mat display;
        float fps_ema = 0.0f;

        if (opt.show_window)
        {
            cv::namedWindow("robocup_infer", cv::WINDOW_NORMAL);
        }

        while (cap.read(frame))
        {
            if (frame.empty())
            {
                continue;
            }

            const auto t0 = cv::getTickCount();

            LetterboxInfo lb = letterbox(frame, letterboxed, opt.input_size);
            ncnn::Mat in = ncnn::Mat::from_pixels(letterboxed.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                  opt.input_size, opt.input_size);
            const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
            in.substract_mean_normalize(nullptr, norm_vals);

            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);

            ncnn::Mat out;
            ex.extract("out0", out);

            std::vector<Detection> detections = decode_detections(
                out, frame.cols, frame.rows, lb, opt.conf_thres, opt.nms_thres);

            int16_t delta_x = 999;
            bool found = false;
            if (!detections.empty())
            {
                const Detection &best = detections.front();
                const float cx = best.box.x + best.box.width * 0.5f;
                delta_x = static_cast<int16_t>(std::round(cx - frame.cols * 0.5f));
                found = true;
            }

            if (found)
            {
                send_delta(serial_fd, delta_x);
            }
            else
            {
                send_delta(serial_fd, 999);
            }

            display = frame.clone();
            draw_detections(display, detections);

            const float instant_fps = static_cast<float>(cv::getTickFrequency() / (cv::getTickCount() - t0));
            if (fps_ema <= 0.0f)
            {
                fps_ema = instant_fps;
            }
            else
            {
                fps_ema = 0.9f * fps_ema + 0.1f * instant_fps;
            }

            std::string fps_text = "FPS: " + cv::format("%.1f", fps_ema);
            cv::putText(display, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            if (found)
            {
                std::string dx_text = "dx: " + std::to_string(delta_x);
                cv::putText(display, dx_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                            0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            if (opt.show_window)
            {
                cv::imshow("robocup_infer", display);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q')
                {
                    break;
                }
            }
        }

        if (serial_fd >= 0)
        {
            close(serial_fd);
        }
        cap.release();
        if (opt.show_window)
        {
            cv::destroyAllWindows();
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
