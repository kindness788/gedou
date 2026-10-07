#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <utility>
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
    int yolo_interval = 3;
    int min_track_points = 6;
    int baudrate = 115200;
    float distance_threshold = 50.0f;
    float conf_thres = 0.25f;
    float nms_thres = 0.45f;
    float min_hsv_ratio = 0.008f;
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

struct TrackingState
{
    bool active = false;
    cv::Rect2f box;
    std::vector<cv::Point2f> points;
    float score = 0.0f;
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
        else if (arg == "--yolo-interval")
        {
            opt.yolo_interval = std::stoi(need_value(arg));
        }
        else if (arg == "--track-points")
        {
            opt.min_track_points = std::stoi(need_value(arg));
        }
        else if (arg == "--hsv-ratio")
        {
            opt.min_hsv_ratio = std::stof(need_value(arg));
        }
        else if (arg == "--baud")
        {
            opt.baudrate = std::stoi(need_value(arg));
        }
        else if (arg == "--distance-threshold")
        {
            opt.distance_threshold = std::stof(need_value(arg));
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
    if (opt.input_size <= 0 || opt.input_size % 32 != 0)
    {
        throw std::runtime_error("--size must be a positive multiple of 32");
    }
    if (opt.yolo_interval < 1)
    {
        throw std::runtime_error("--yolo-interval must be at least 1");
    }
    if (opt.min_track_points < 3)
    {
        throw std::runtime_error("--track-points must be at least 3");
    }
    if (opt.min_hsv_ratio < 0.0f || opt.min_hsv_ratio > 1.0f)
    {
        throw std::runtime_error("--hsv-ratio must be between 0 and 1");
    }
    if (!std::isfinite(opt.distance_threshold) || opt.distance_threshold <= 0.0f)
    {
        throw std::runtime_error("--distance-threshold must be a positive finite number");
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

// Bit 0: debuff present; bit 1: buff present; bits 2/3: debuff/buff left;
// bit 4: priority target within distance threshold; bits 5..7: reserved.
static uint8_t encode_target_status(const Detection *debuff,
                                    const Detection *buff,
                                    int frame_width,
                                    float distance_threshold)
{
    uint8_t status = 0;
    if (debuff != nullptr)
    {
        status |= 0x01;
        const float center_x = debuff->box.x + debuff->box.width * 0.5f;
        if (center_x < frame_width * 0.5f)
        {
            status |= 0x04;
        }
    }
    if (buff != nullptr)
    {
        status |= 0x02;
        const float center_x = buff->box.x + buff->box.width * 0.5f;
        if (center_x < frame_width * 0.5f)
        {
            status |= 0x08;
        }
    }
    const Detection *priority = debuff != nullptr ? debuff : buff;
    if (priority != nullptr && priority->box.width > 0.0f &&
        5000.0f / priority->box.width <= distance_threshold)
    {
        status |= 0x10;
    }
    return status;
}

static void send_target_status(int fd, uint8_t status)
{
    if (fd >= 0)
    {
        (void)write(fd, &status, sizeof(status));
    }
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

static cv::Rect clip_box(const cv::Rect2f &box, const cv::Size &image_size)
{
    const int x1 = std::max(0, static_cast<int>(std::floor(box.x)));
    const int y1 = std::max(0, static_cast<int>(std::floor(box.y)));
    const int x2 = std::min(image_size.width, static_cast<int>(std::ceil(box.x + box.width)));
    const int y2 = std::min(image_size.height, static_cast<int>(std::ceil(box.y + box.height)));
    if (x2 <= x1 || y2 <= y1)
    {
        return cv::Rect();
    }
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

static cv::Mat make_buff_hsv_mask(const cv::Mat &frame,
                                  const cv::Rect &roi,
                                  float *yellow_green_ratio = nullptr,
                                  float *blue_ratio = nullptr)
{
    cv::Mat hsv;
    cv::cvtColor(frame(roi), hsv, cv::COLOR_BGR2HSV);

    // Gain blocks contain yellow/green lightning and dark-blue rings. OpenCV
    // stores hue in [0, 179], so these deliberately broad ranges tolerate
    // exposure changes while the ROI keeps the search local.
    cv::Mat yellow_green;
    cv::Mat blue;
    cv::inRange(hsv, cv::Scalar(20, 55, 40), cv::Scalar(90, 255, 255), yellow_green);
    cv::inRange(hsv, cv::Scalar(95, 55, 25), cv::Scalar(140, 255, 255), blue);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(yellow_green, yellow_green, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(blue, blue, cv::MORPH_OPEN, kernel);
    const float area = static_cast<float>(roi.area());
    if (yellow_green_ratio != nullptr)
    {
        *yellow_green_ratio = static_cast<float>(cv::countNonZero(yellow_green)) / area;
    }
    if (blue_ratio != nullptr)
    {
        *blue_ratio = static_cast<float>(cv::countNonZero(blue)) / area;
    }

    cv::Mat mask = yellow_green | blue;
    cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 1);
    return mask;
}

static bool buff_hsv_is_valid(const cv::Mat &frame,
                              const cv::Rect2f &box,
                              float min_hsv_ratio)
{
    const cv::Rect roi = clip_box(box, frame.size());
    if (roi.area() <= 0)
    {
        return false;
    }
    float yellow_green_ratio = 0.0f;
    float blue_ratio = 0.0f;
    const cv::Mat mask = make_buff_hsv_mask(frame, roi, &yellow_green_ratio, &blue_ratio);
    const float total_ratio = static_cast<float>(cv::countNonZero(mask)) /
                              static_cast<float>(roi.area());

    // Requiring evidence from both parts of the gain logo prevents a tracker
    // that has drifted onto the green arena floor from being considered valid.
    return total_ratio >= min_hsv_ratio &&
           yellow_green_ratio >= min_hsv_ratio * 0.15f &&
           blue_ratio >= min_hsv_ratio * 0.10f;
}

static std::vector<cv::Point2f> find_tracking_points(const cv::Mat &gray,
                                                      const cv::Mat &frame,
                                                      const cv::Rect2f &box,
                                                      int min_points)
{
    const cv::Rect roi = clip_box(box, frame.size());
    if (roi.width < 4 || roi.height < 4)
    {
        return {};
    }

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    cv::Mat local_mask = make_buff_hsv_mask(frame, roi);
    local_mask.copyTo(mask(roi));

    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(gray, points, 80, 0.01, 4.0, mask, 3, false, 0.04);

    // Uniform colored areas may not contain enough corners. In that case use
    // grayscale corners from the same YOLO ROI; HSV is still checked on every
    // tracking update to reject points that drift onto the background.
    if (static_cast<int>(points.size()) < min_points)
    {
        mask.setTo(0);
        mask(roi).setTo(255);
        cv::goodFeaturesToTrack(gray, points, 80, 0.01, 4.0, mask, 3, false, 0.04);
    }
    return points;
}

static bool initialize_tracker(TrackingState &tracker,
                               const cv::Mat &gray,
                               const cv::Mat &frame,
                               const Detection &target,
                               int min_points)
{
    tracker = TrackingState{};
    const cv::Rect roi = clip_box(target.box, frame.size());
    if (roi.width < 4 || roi.height < 4)
    {
        return false;
    }

    tracker.box = cv::Rect2f(static_cast<float>(roi.x), static_cast<float>(roi.y),
                             static_cast<float>(roi.width), static_cast<float>(roi.height));
    tracker.points = find_tracking_points(gray, frame, tracker.box, min_points);
    tracker.score = target.score;
    tracker.active = static_cast<int>(tracker.points.size()) >= min_points;
    return tracker.active;
}

static float median_value(std::vector<float> values)
{
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    float result = values[middle];
    if (values.size() % 2 == 0)
    {
        const auto lower = std::max_element(values.begin(), values.begin() + middle);
        result = (*lower + result) * 0.5f;
    }
    return result;
}

static bool update_tracker(TrackingState &tracker,
                           const cv::Mat &previous_gray,
                           const cv::Mat &gray,
                           const cv::Mat &frame,
                           int min_points,
                           float min_hsv_ratio)
{
    if (!tracker.active || previous_gray.empty() ||
        static_cast<int>(tracker.points.size()) < min_points)
    {
        tracker.active = false;
        return false;
    }

    std::vector<cv::Point2f> next_points;
    std::vector<uchar> status;
    std::vector<float> errors;
    cv::calcOpticalFlowPyrLK(previous_gray, gray, tracker.points, next_points,
                             status, errors, cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                              20, 0.03));

    std::vector<cv::Point2f> candidates;
    std::vector<float> dxs;
    std::vector<float> dys;
    for (size_t i = 0; i < next_points.size(); ++i)
    {
        const cv::Point2f &point = next_points[i];
        if (!status[i] || errors[i] > 30.0f || point.x < 0.0f || point.y < 0.0f ||
            point.x >= frame.cols || point.y >= frame.rows)
        {
            continue;
        }
        candidates.push_back(point);
        dxs.push_back(point.x - tracker.points[i].x);
        dys.push_back(point.y - tracker.points[i].y);
    }

    if (static_cast<int>(candidates.size()) < min_points)
    {
        tracker.active = false;
        return false;
    }

    const float median_dx = median_value(dxs);
    const float median_dy = median_value(dys);
    const float max_residual = std::max(2.5f, 0.12f * std::min(tracker.box.width, tracker.box.height));
    std::vector<cv::Point2f> consistent_points;
    std::vector<float> consistent_dxs;
    std::vector<float> consistent_dys;
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        const float residual = std::hypot(dxs[i] - median_dx, dys[i] - median_dy);
        if (residual <= max_residual)
        {
            consistent_points.push_back(candidates[i]);
            consistent_dxs.push_back(dxs[i]);
            consistent_dys.push_back(dys[i]);
        }
    }

    if (static_cast<int>(consistent_points.size()) < min_points)
    {
        tracker.active = false;
        return false;
    }

    tracker.box.x += median_value(consistent_dxs);
    tracker.box.y += median_value(consistent_dys);
    const cv::Rect clipped = clip_box(tracker.box, frame.size());
    if (clipped.width < 4 || clipped.height < 4)
    {
        tracker.active = false;
        return false;
    }
    tracker.box = cv::Rect2f(static_cast<float>(clipped.x), static_cast<float>(clipped.y),
                             static_cast<float>(clipped.width), static_cast<float>(clipped.height));

    if (!buff_hsv_is_valid(frame, tracker.box, min_hsv_ratio))
    {
        tracker.active = false;
        return false;
    }

    std::vector<cv::Point2f> refreshed = find_tracking_points(gray, frame, tracker.box, min_points);
    tracker.points = static_cast<int>(refreshed.size()) >= min_points
                         ? std::move(refreshed)
                         : std::move(consistent_points);
    tracker.active = static_cast<int>(tracker.points.size()) >= min_points;
    return tracker.active;
}

static bool select_largest_class(const std::vector<Detection> &detections,
                                int class_id,
                                Detection &target)
{
    float largest_area = 0.0f;
    bool found = false;
    for (const auto &det : detections)
    {
        if (det.class_id != class_id)
        {
            continue;
        }
        const float area = det.box.width * det.box.height;
        if (det.box.width > 0.0f && area > largest_area)
        {
            largest_area = area;
            target = det;
            found = true;
        }
    }
    return found;
}

static std::vector<Detection> run_yolo(ncnn::Net &net,
                                        const cv::Mat &frame,
                                        int input_size,
                                        float conf_thres,
                                        float nms_thres)
{
    cv::Mat prepared;
    const LetterboxInfo lb = letterbox(frame, prepared, input_size);
    ncnn::Mat in = ncnn::Mat::from_pixels(prepared.data, ncnn::Mat::PIXEL_BGR2RGB,
                                          input_size, input_size);
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in.substract_mean_normalize(nullptr, norm_vals);

    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("in0", in) != 0)
    {
        throw std::runtime_error("Failed to set NCNN input tensor 'in0'");
    }

    ncnn::Mat out;
    if (extractor.extract("out0", out) != 0)
    {
        throw std::runtime_error("Failed to extract NCNN output tensor 'out0'");
    }
    return decode_detections(out, frame.cols, frame.rows, lb, conf_thres, nms_thres);
}

static std::string class_name(int class_id)
{
    switch (class_id)
    {
    case 0:
        return "buff_block";
    case 1:
        return "debuff_block";
    default:
        return "unknown";
    }
}

static void draw_detection(cv::Mat &image, const Detection &det)
{
    // OpenCV uses BGR: class 0 (buff) is green, class 1 (debuff) is red.
    cv::Scalar color(200, 200, 200);
    if (det.class_id == 0)
        color = cv::Scalar(0, 255, 0);
    else if (det.class_id == 1)
        color = cv::Scalar(0, 0, 255);

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

static void draw_detections(cv::Mat &image, const std::vector<Detection> &detections)
{
    for (const auto &det : detections)
    {
        draw_detection(image, det);
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
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 320);
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
        cv::Mat previous_gray;
        cv::Mat display;
        TrackingState tracker;
        uint64_t frame_index = 0;
        bool debuff_priority = false;
        int debuff_miss_frames = 0;
        constexpr int debuff_exit_miss_frames = 3;
        float yolo_fps_ema = 0.0f;
        float loop_fps_ema = 0.0f;

        std::cout << "YOLO interval: " << opt.yolo_interval
                  << " frames, minimum tracking points: " << opt.min_track_points
                  << ", minimum HSV ratio: " << opt.min_hsv_ratio << std::endl;

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

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // Track on every frame first. Every Nth frame YOLO corrects the
            // accumulated drift; when optical flow or HSV validation fails,
            // YOLO is called immediately instead of waiting for the schedule.
            bool tracking_ok = false;
            if (tracker.active && !debuff_priority)
            {
                tracking_ok = update_tracker(tracker, previous_gray, gray, frame,
                                             opt.min_track_points, opt.min_hsv_ratio);
            }

            const bool scheduled_yolo = (frame_index % static_cast<uint64_t>(opt.yolo_interval)) == 0;
            const bool should_run_yolo = debuff_priority || scheduled_yolo || !tracking_ok;
            bool yolo_ran = false;
            bool target_from_yolo = false;
            std::vector<Detection> detections;
            Detection target;
            uint8_t target_type = 0x00;

            if (should_run_yolo)
            {
                const int64_t yolo_start = cv::getTickCount();
                detections = run_yolo(net, frame, opt.input_size,
                                      opt.conf_thres, opt.nms_thres);
                const float yolo_fps = static_cast<float>(
                    cv::getTickFrequency() / (cv::getTickCount() - yolo_start));
                yolo_fps_ema = yolo_fps_ema <= 0.0f
                                   ? yolo_fps
                                   : 0.9f * yolo_fps_ema + 0.1f * yolo_fps;
                yolo_ran = true;

                Detection debuff_target;
                if (select_largest_class(detections, 1, debuff_target))
                {
                    // A detected debuff always wins, including when a gain
                    // block is also visible. Never continue gain optical flow.
                    target = debuff_target;
                    target_type = 0x02;
                    target_from_yolo = true;
                    tracker = TrackingState{};
                    debuff_priority = true;
                    debuff_miss_frames = 0;
                }
                else if (debuff_priority)
                {
                    // In debuff mode YOLO runs every frame. Require several
                    // consecutive misses before allowing gain tracking again.
                    ++debuff_miss_frames;
                    tracker = TrackingState{};
                    if (debuff_miss_frames >= debuff_exit_miss_frames)
                    {
                        debuff_priority = false;
                        debuff_miss_frames = 0;
                        Detection gain_target;
                        if (select_largest_class(detections, 0, gain_target))
                        {
                            target = gain_target;
                            target_type = 0x01;
                            target_from_yolo = true;
                            initialize_tracker(tracker, gray, frame, target, opt.min_track_points);
                        }
                    }
                }
                else
                {
                    Detection gain_target;
                    if (select_largest_class(detections, 0, gain_target))
                    {
                        target = gain_target;
                        target_type = 0x01;
                        target_from_yolo = true;
                        // Use the YOLO detection now, then initialize tracking
                        // for the following camera frame.
                        initialize_tracker(tracker, gray, frame, target, opt.min_track_points);
                    }
                    else if (tracking_ok)
                    {
                        target.box = tracker.box;
                        target.score = tracker.score;
                        target.class_id = 0;
                        target_type = 0x01;
                    }
                    else
                    {
                        tracker = TrackingState{};
                    }
                }
            }
            else
            {
                target.box = tracker.box;
                target.score = tracker.score;
                target.class_id = 0;
                target_type = 0x01;
            }

            Detection debuff_status_target;
            Detection buff_status_target;
            bool has_debuff = yolo_ran &&
                              select_largest_class(detections, 1, debuff_status_target);
            bool has_buff = yolo_ran &&
                            select_largest_class(detections, 0, buff_status_target);
            if (target_type == 0x01 && !has_buff)
            {
                buff_status_target = target;
                has_buff = true;
            }

            // Presence and direction are independent for the two classes.
            // If both are visible, the distance flag follows debuff priority.
            const uint8_t status = encode_target_status(
                has_debuff ? &debuff_status_target : nullptr,
                has_buff ? &buff_status_target : nullptr,
                frame.cols, opt.distance_threshold);
            send_target_status(serial_fd, status);

            display = frame.clone();
            draw_detections(display, detections);

            if (target_type != 0x00)
            {
                const cv::Scalar target_color = target_type == 0x02
                                                    ? cv::Scalar(0, 0, 255)
                                                    : (target_from_yolo
                                                           ? cv::Scalar(0, 255, 0)
                                                           : cv::Scalar(255, 255, 0));
                cv::rectangle(display, target.box, target_color, 3);
                const std::string source = target_type == 0x02
                                               ? "DEBUFF PRIORITY"
                                               : (target_from_yolo ? "GAIN YOLO" : "GAIN FLOW");
                const int label_y = std::max(18, static_cast<int>(target.box.y) - 6);
                cv::putText(display, source, cv::Point(std::max(0, static_cast<int>(target.box.x)), label_y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, target_color, 2, cv::LINE_AA);
            }

            std::string fps_text = "YOLO FPS: " + cv::format("%.1f", yolo_fps_ema);
            cv::putText(display, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            if (loop_fps_ema > 0.0f)
            {
                std::string loop_text = "Loop FPS: " + cv::format("%.1f", loop_fps_ema);
                cv::putText(display, loop_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                            0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            const std::string status_text = cv::format("Serial: 0x%02X", status);
            cv::putText(display, status_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            const std::string mode_text = debuff_priority
                                              ? (target_type == 0x02 ? "Mode: DEBUFF YOLO" : "Mode: DEBUFF SEARCH")
                                              : (yolo_ran ? "Mode: YOLO" : "Mode: GAIN FLOW+HSV");
            cv::putText(display, mode_text, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX,
                        0.65, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

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

            gray.copyTo(previous_gray);
            ++frame_index;
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
