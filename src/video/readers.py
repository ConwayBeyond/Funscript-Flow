"""Video reader classes for frame extraction"""

import cv2
import numpy as np
import threading
import concurrent.futures
import gc
from queue import Queue, Empty
from typing import List, Optional, Dict


class AsyncVideoReader:
    """High-performance async video reader with frame buffering and parallel decoding."""
    
    def __init__(self, video_path: str, width: Optional[int] = None, height: Optional[int] = None, 
                 num_threads: int = 4, buffer_size: int = 100):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.num_threads = min(num_threads, 4)  # Limit decoder threads
        self.buffer_size = buffer_size
        
        # Test video can be opened
        test_cap = cv2.VideoCapture(video_path)
        if not test_cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        self.total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = test_cap.get(cv2.CAP_PROP_FPS)
        self.orig_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        test_cap.release()
        
        # Frame buffer and memory pool
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.frame_pool = Queue()
        
        # Pre-allocate frame buffers
        for _ in range(buffer_size):
            if self.width and self.height:
                frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
            else:
                frame = np.empty((self.orig_height, self.orig_width, 3), dtype=np.uint8)
            self.frame_pool.put(frame)
        
        # Decoder threads and state
        self.decoders = []
        self.decoder_locks = []
        self.stop_event = threading.Event()
        self.prefetch_thread = None
        
        # Initialize multiple decoders
        for i in range(self.num_threads):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal internal buffering
            self.decoders.append(cap)
            self.decoder_locks.append(threading.Lock())
    
    def __len__(self):
        return self.total_frames
    
    def get_avg_fps(self):
        return self.fps
    
    def _get_frame_buffer(self) -> np.ndarray:
        """Get a frame buffer from the pool or allocate a new one."""
        try:
            return self.frame_pool.get_nowait()
        except Empty:
            if self.width and self.height:
                return np.empty((self.height, self.width, 3), dtype=np.uint8)
            else:
                return np.empty((self.orig_height, self.orig_width, 3), dtype=np.uint8)
    
    def _return_frame_buffer(self, frame: np.ndarray):
        """Return a frame buffer to the pool."""
        try:
            self.frame_pool.put_nowait(frame)
        except:
            pass  # Pool is full, let GC handle it
    
    def _decode_frame(self, idx: int, decoder_id: int) -> Optional[np.ndarray]:
        """Decode a single frame using the specified decoder."""
        with self.decoder_locks[decoder_id]:
            cap = self.decoders[decoder_id]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB in-place
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
                
                # Resize if needed
                if self.width and self.height and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                return frame
            return None
    
    def _prefetch_worker(self, indices: List[int]):
        """Worker thread that prefetches frames into the buffer."""
        decoder_id = 0
        
        for idx in indices:
            if self.stop_event.is_set():
                break
            
            frame = self._decode_frame(idx, decoder_id)
            if frame is not None:
                self.frame_buffer.put((idx, frame))
            
            # Round-robin through decoders
            decoder_id = (decoder_id + 1) % self.num_threads
    
    def get_batch(self, indices: List[int]) -> np.ndarray:
        """Get a batch of frames by indices with parallel decoding."""
        frames = [None] * len(indices)
        
        # Start prefetch thread
        self.stop_event.clear()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(indices,))
        self.prefetch_thread.start()
        
        # Collect frames from buffer
        received = 0
        timeout = 10.0  # seconds
        
        while received < len(indices):
            try:
                idx, frame = self.frame_buffer.get(timeout=timeout)
                # Find position in indices
                try:
                    pos = indices.index(idx)
                    frames[pos] = frame
                    received += 1
                except ValueError:
                    # Frame not in requested indices, return to pool
                    self._return_frame_buffer(frame)
            except Empty:
                break
        
        # Stop prefetch and wait
        self.stop_event.set()
        if self.prefetch_thread:
            self.prefetch_thread.join()
        
        # Fill any missing frames with black
        for i, frame in enumerate(frames):
            if frame is None:
                if self.width and self.height:
                    frames[i] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frames[i] = np.zeros((self.orig_height, self.orig_width, 3), dtype=np.uint8)
        
        return np.array(frames)
    
    def get_batch_parallel(self, indices: List[int]) -> np.ndarray:
        """Alternative parallel batch reading using thread pool."""
        frames = [None] * len(indices)
        
        def decode_with_id(args):
            idx, pos, decoder_id = args
            frame = self._decode_frame(idx, decoder_id)
            return pos, frame
        
        # Distribute indices across decoders
        tasks = []
        for i, idx in enumerate(indices):
            decoder_id = i % self.num_threads
            tasks.append((idx, i, decoder_id))
        
        # Decode in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(decode_with_id, task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    pos, frame = future.result()
                    frames[pos] = frame
                except Exception as e:
                    pass
        
        # Fill missing frames
        for i, frame in enumerate(frames):
            if frame is None:
                if self.width and self.height:
                    frames[i] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frames[i] = np.zeros((self.orig_height, self.orig_width, 3), dtype=np.uint8)
        
        return np.array(frames)
    
    def __del__(self):
        self.stop_event.set()
        if self.prefetch_thread:
            self.prefetch_thread.join()
        
        for cap in self.decoders:
            if cap:
                cap.release()


class VideoReaderCV:
    """OpenCV VideoReader wrapper to replace decord."""
    
    def __init__(self, video_path, width=None, height=None, num_threads=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = width
        self.height = height
        
        # Get original dimensions
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def __len__(self):
        return self.total_frames
    
    def get_avg_fps(self):
        return self.fps
    
    def get_batch(self, indices):
        frames = []
        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize if needed
                if self.width is not None and self.height is not None:
                    frame = cv2.resize(frame, (self.width, self.height))
                frames.append(frame)
            else:
                # If frame reading fails, create a black frame
                if self.width is not None and self.height is not None:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((self.orig_height, self.orig_width, 3), dtype=np.uint8)
                frames.append(frame)
        
        # Return as numpy array to mimic decord's asnumpy() behavior
        return np.array(frames)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()