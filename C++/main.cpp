#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <opencv2/opencv.hpp>
#include <net.h>

struct Options
{
    std::string model_param = "model.ncnn.param";
    std::string model_bin = "model.ncnn.bin";
    std::string serial_port = "/dev/ttyUSB0";
    std::string remote_ip = "192.168.139.200";
    int remote_port = 8888;
    int udp_quality = 60;
    int udp_fps = 15;
    int udp_chunk_size = 1200;
    int camera_index = 0;
    int input_size = 320;
    int baudrate = 115200;
    float conf_thres = 0.25f;
    float nms_thres = 0.45f;
    bool enable_udp = true;
    bool show_window = false;
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
        else if (arg == "--ip")
        {
            opt.remote_ip = need_value(arg);
        }
        else if (arg == "--port")
        {
            opt.remote_port = std::stoi(need_value(arg));
        }
        else if (arg == "--udp-quality")
        {
            opt.udp_quality = std::stoi(need_value(arg));
        }
        else if (arg == "--udp-fps")
        {
            opt.udp_fps = std::stoi(need_value(arg));
        }
        else if (arg == "--udp-chunk")
        {
            opt.udp_chunk_size = std::stoi(need_value(arg));
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
        else if (arg == "--no-udp")
        {
            opt.enable_udp = false;
        }
        else if (arg == "--show")
        {
            opt.show_window = true;
        }
        else if (arg == "--no-show")
        {
            opt.show_window = false;
        }
    }
    return opt;
}

static void validate_udp_options(const Options &opt)
{
    if (opt.remote_port < 1 || opt.remote_port > 65535)
    {
        throw std::runtime_error("UDP port must be between 1 and 65535");
    }
    if (opt.udp_quality < 1 || opt.udp_quality > 100)
    {
        throw std::runtime_error("UDP JPEG quality must be between 1 and 100");
    }
    if (opt.udp_fps < 0)
    {
        throw std::runtime_error("UDP FPS must be zero or greater");
    }
    if (opt.udp_chunk_size < 256 || opt.udp_chunk_size > 1400)
    {
        throw std::runtime_error("UDP chunk size must be between 256 and 1400 bytes");
    }
}

static bool create_udp_target(const Options &opt, int &socket_fd, sockaddr_in &target)
{
    socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0)
    {
        std::cerr << "Failed to create UDP socket: " << std::strerror(errno) << std::endl;
        return false;
    }

    target = {};
    target.sin_family = AF_INET;
    target.sin_port = htons(static_cast<uint16_t>(opt.remote_port));
    if (inet_pton(AF_INET, opt.remote_ip.c_str(), &target.sin_addr) != 1)
    {
        std::cerr << "Invalid UDP destination IP: " << opt.remote_ip << std::endl;
        close(socket_fd);
        socket_fd = -1;
        return false;
    }

    return true;
}

static bool send_jpeg_udp(int socket_fd,
                          const sockaddr_in &target,
                          const std::vector<uchar> &jpeg,
                          uint32_t frame_id,
                          int chunk_size)
{
    // Keep datagrams below the usual Wi-Fi MTU. The receiver uses this header
    // to discard incomplete frames and reassemble complete JPEGs.
    constexpr size_t header_size = 12;
    if (jpeg.empty() || chunk_size < static_cast<int>(header_size))
    {
        return false;
    }

    const size_t payload_size = static_cast<size_t>(chunk_size) - header_size;
    const size_t chunk_count = (jpeg.size() + payload_size - 1) / payload_size;
    if (chunk_count == 0 || chunk_count > std::numeric_limits<uint16_t>::max())
    {
        return false;
    }

    std::vector<uint8_t> packet(header_size + payload_size);
    packet[0] = 'G';
    packet[1] = 'D';
    packet[2] = 'U';
    packet[3] = '1';

    const uint32_t network_frame_id = htonl(frame_id);
    std::memcpy(packet.data() + 4, &network_frame_id, sizeof(network_frame_id));

    const uint16_t network_chunk_count = htons(static_cast<uint16_t>(chunk_count));
    std::memcpy(packet.data() + 8, &network_chunk_count, sizeof(network_chunk_count));

    for (size_t chunk_id = 0; chunk_id < chunk_count; ++chunk_id)
    {
        const size_t offset = chunk_id * payload_size;
        const size_t bytes = std::min(payload_size, jpeg.size() - offset);
        const uint16_t network_chunk_id = htons(static_cast<uint16_t>(chunk_id));
        std::memcpy(packet.data() + 10, &network_chunk_id, sizeof(network_chunk_id));
        std::memcpy(packet.data() + header_size, jpeg.data() + offset, bytes);

        const ssize_t sent = sendto(socket_fd,
                                    packet.data(),
                                    header_size + bytes,
                                    0,
                                    reinterpret_cast<const sockaddr *>(&target),
                                    sizeof(target));
        if (sent != static_cast<ssize_t>(header_size + bytes))
        {
            return false;
        }
    }
    return true;
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

static void send_target_data(int fd, bool found, int16_t delta_x, int16_t distance)
{
    if (fd < 0)
    {
        return;
    }

    uint8_t packet[8];
    packet[0] = 0xAA;
    packet[1] = found ? 0x01 : 0x00;
    packet[2] = static_cast<uint8_t>((delta_x >> 8) & 0xFF);
    packet[3] = static_cast<uint8_t>(delta_x & 0xFF);
    packet[4] = static_cast<uint8_t>((distance >> 8) & 0xFF);
    packet[5] = static_cast<uint8_t>(distance & 0xFF);
    packet[6] = static_cast<uint8_t>((packet[1] + packet[2] + packet[3] +
                                      packet[4] + packet[5]) &
                                     0xFF);
    packet[7] = 0xBB;
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

static std::vector<Detection> decode_detections(const ncnn::Mat &out,
                                                int img_w,
                                                int img_h,
                                                const LetterboxInfo &lb,
                                                float conf_thres,
                                                float nms_thres)
{
    if (out.dims != 2)
    {
        throw std::runtime_error("Unexpected NCNN output: expected a 2-D tensor");
    }

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    const int num_candidates = (out.h <= out.w) ? out.w : out.h;
    const int num_attributes = (out.h <= out.w) ? out.h : out.w;
    const bool normal_layout = (out.h <= out.w);
    const int num_classes = num_attributes - 4;
    if (num_classes < 2)
    {
        throw std::runtime_error("Unexpected NCNN output: fewer than two class scores");
    }

    auto value_at = [&](int attribute, int candidate) -> float
    {
        return normal_layout ? out.row(attribute)[candidate]
                             : out.row(candidate)[attribute];
    };

    for (int i = 0; i < num_candidates; ++i)
    {
        const float cx = value_at(0, i);
        const float cy = value_at(1, i);
        const float w = value_at(2, i);
        const float h = value_at(3, i);

        int class_id = 0;
        float score = value_at(4, i);
        for (int c = 1; c < num_classes; ++c)
        {
            const float class_score = value_at(4 + c, i);
            if (class_score > score)
            {
                score = class_score;
                class_id = c;
            }
        }

        if (score < conf_thres)
        {
            continue;
        }

        // Ultralytics' non-end-to-end NCNN export emits cx, cy, width, height.
        // Never guess xyxy from the values: that breaks boxes near the top/left edge.
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

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

        boxes.emplace_back(static_cast<double>(x1), static_cast<double>(y1),
                           static_cast<double>(bw), static_cast<double>(bh));
        scores.emplace_back(score);
        class_ids.emplace_back(class_id);
    }

    // Run NMS per class so an overlapping buff and debuff do not suppress each other.
    std::vector<int> keep;
    for (int class_id = 0; class_id < num_classes; ++class_id)
    {
        std::vector<cv::Rect2d> class_boxes;
        std::vector<float> class_scores;
        std::vector<int> source_indices;
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (class_ids[i] == class_id)
            {
                class_boxes.push_back(boxes[i]);
                class_scores.push_back(scores[i]);
                source_indices.push_back(static_cast<int>(i));
            }
        }

        std::vector<int> class_keep;
        cv::dnn::NMSBoxes(class_boxes, class_scores, conf_thres, nms_thres, class_keep);
        for (int idx : class_keep)
        {
            keep.push_back(source_indices[idx]);
        }
    }

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
        if (opt.enable_udp)
        {
            validate_udp_options(opt);
        }

        ncnn::Net net;
        net.opt.num_threads = std::max(1, cv::getNumberOfCPUs() - 1);
        if (net.load_param(opt.model_param.c_str()) != 0)
        {
            throw std::runtime_error("Failed to load NCNN param: " + opt.model_param);
        }
        if (net.load_model(opt.model_bin.c_str()) != 0)
        {
            throw std::runtime_error("Failed to load NCNN weights: " + opt.model_bin);
        }

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

        int udp_sock = -1;
        sockaddr_in udp_target{};
        if (opt.enable_udp && !create_udp_target(opt, udp_sock, udp_target))
        {
            if (serial_fd >= 0)
            {
                close(serial_fd);
            }
            cap.release();
            return 1;
        }

        if (opt.enable_udp)
        {
            std::cout << "UDP video target: " << opt.remote_ip << ":" << opt.remote_port << std::endl;
        }
        uint32_t udp_frame_id = 0;
        const int64_t udp_interval_ticks = opt.udp_fps > 0
                                               ? static_cast<int64_t>(cv::getTickFrequency() / opt.udp_fps)
                                               : 0;
        int64_t last_udp_tick = 0;

        cv::Mat frame;
        cv::Mat letterboxed;
        cv::Mat display;
        float infer_fps_ema = 0.0f;
        float loop_fps_ema = 0.0f;

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
            if (ex.input("in0", in) != 0)
            {
                throw std::runtime_error("Failed to set NCNN input tensor 'in0'");
            }

            ncnn::Mat out;
            if (ex.extract("out0", out) != 0)
            {
                throw std::runtime_error("Failed to extract NCNN output tensor 'out0'");
            }

            std::vector<Detection> detections = decode_detections(
                out, frame.cols, frame.rows, lb, opt.conf_thres, opt.nms_thres);

            int16_t delta_x = 0;
            int16_t distance = 0;
            bool found = false;
            float max_area = 0.0f;

            // Only buff_block (class 0) is a valid target. Among those targets,
            // the largest box is treated as the nearest physical block.
            for (const auto &det : detections)
            {
                if (det.class_id != 0)
                {
                    continue;
                }

                const float area = det.box.width * det.box.height;
                if (area <= max_area || det.box.width <= 0.0f)
                {
                    continue;
                }

                max_area = area;
                const float cx = det.box.x + det.box.width * 0.5f;
                delta_x = static_cast<int16_t>(std::round(cx - frame.cols * 0.5f));
                distance = static_cast<int16_t>(std::round(5000.0f / det.box.width));
                found = true;
            }
            // 串口通信发送
            send_target_data(serial_fd, found, delta_x, distance);

            display = frame.clone();
            draw_detections(display, detections);

            const float instant_fps = static_cast<float>(cv::getTickFrequency() / (cv::getTickCount() - t0));
            if (infer_fps_ema <= 0.0f)
            {
                infer_fps_ema = instant_fps;
            }
            else
            {
                infer_fps_ema = 0.9f * infer_fps_ema + 0.1f * instant_fps;
            }

            std::string fps_text = "Infer FPS: " + cv::format("%.1f", infer_fps_ema);
            cv::putText(display, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            if (loop_fps_ema > 0.0f)
            {
                std::string loop_text = "Loop FPS: " + cv::format("%.1f", loop_fps_ema);
                cv::putText(display, loop_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                            0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            if (found)
            {
                std::string dx_text = "dx: " + std::to_string(delta_x);
                cv::putText(display, dx_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX,
                            0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            const int64_t now_tick = cv::getTickCount();
            const bool send_udp_frame = opt.enable_udp &&
                                        (opt.udp_fps == 0 ||
                                         last_udp_tick == 0 ||
                                         now_tick - last_udp_tick >= udp_interval_ticks);
            if (send_udp_frame)
            {
                std::vector<uchar> jpeg;
                const std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, opt.udp_quality};
                if (cv::imencode(".jpg", display, jpeg, jpeg_params) &&
                    send_jpeg_udp(udp_sock, udp_target, jpeg, udp_frame_id++, opt.udp_chunk_size))
                {
                    last_udp_tick = now_tick;
                }
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

            const float loop_fps = static_cast<float>(
                cv::getTickFrequency() / (cv::getTickCount() - t0));
            if (loop_fps_ema <= 0.0f)
            {
                loop_fps_ema = loop_fps;
            }
            else
            {
                loop_fps_ema = 0.9f * loop_fps_ema + 0.1f * loop_fps;
            }
        }

        if (serial_fd >= 0)
        {
            close(serial_fd);
        }
        if (udp_sock >= 0)
        {
            close(udp_sock);
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
