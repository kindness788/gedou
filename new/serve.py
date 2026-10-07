from flask import Flask, render_template
from flask_socketio import SocketIO
import socket
import struct  # 新增：用于解析 C++ 发过来的二进制包头

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UDP_IP = "0.0.0.0"
UDP_PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

def udp_listener():
    print(f"UDP 接收端已启动，正在监听端口 {UDP_PORT}...")
    
    frame_buffer = {}
    current_frame_id = -1
    
    while True:
        data, addr = sock.recvfrom(65535)
        
        # 检查是否是我们在 C++ 端定义的 GDU1 数据包头 (12字节)
        if len(data) >= 12 and data[:4] == b'GDU1':
            # 解析包头：>IHH 代表大端序的 uint32 (frame_id), uint16 (chunk_count), uint16 (chunk_id)
            frame_id, chunk_count, chunk_id = struct.unpack('>IHH', data[4:12])
            payload = data[12:]  # 剥离包头，取出真正的 JPEG 图像数据
            
            # 如果收到了新的一帧，清空之前的分片缓存
            if frame_id != current_frame_id:
                frame_buffer = {}
                current_frame_id = frame_id
                
            # 将当前分片存入字典
            frame_buffer[chunk_id] = payload
            
            # 如果这一帧的所有分片都收齐了
            if len(frame_buffer) == chunk_count:
                # 按照 chunk_id 的顺序将所有分片拼接成完整的 JPEG
                full_jpeg = b''.join(frame_buffer[i] for i in range(chunk_count) if i in frame_buffer)
                
                # 将拼接好的完整图片通过 WebSocket 发送给前端
                socketio.emit('video_frame', {'image': full_jpeg})
        else:
            # 兼容处理：如果没有 GDU1 包头（比如发来了完整未分片的图），直接转发
            socketio.emit('video_frame', {'image': data})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    import threading
    t = threading.Thread(target=udp_listener, daemon=True)
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000)