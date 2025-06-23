Funscript Flow uses computer vision to automatically generate decent funscript files synchronized to any video. It works in batch mode, so you can just select a folder and hit "Run" (it can get through a hundred gigs per day or so, on my setup).

**How to Get it**

The easiest way is to [download it here](https://github.com/Funscript-Flow/Funscript-Flow/releases). It's a packaged executable that should just work on Windows.

**Running from Source**

**Prerequisites:** Python 3.13 or higher

**Using uv (recommended):**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and run
uv run FunscriptFlow.pyw
```

**Using pip:**
```bash
pip install numpy opencv-python
python FunscriptFlow.pyw
```

**Building Standalone Executables:**

First install build dependencies:
```bash
uv sync --extra build
```

**Windows:**
```bash
uv run nuitka FunscriptFlow.pyw --standalone --enable-plugin=tk-inter --windows-disable-console --windows-icon-from-ico=icon.ico
```

**macOS:**
```bash
uv run nuitka FunscriptFlow.pyw --standalone --enable-plugin=tk-inter --macos-create-app-bundle --macos-app-icon=icon.png
```

**Linux:**
```bash
uv run nuitka FunscriptFlow.pyw --standalone --enable-plugin=tk-inter --linux-icon=icon.png
```

**Features**
* Completely automatic motion-tracked funscript generation for any video.
* Easy to use — Just download it and point it at your 3-terabyte “Taxes” folder.
* Runs on your computer without uploading anything anywhere.
* Generates scripts for entire libraries, fast (usually faster than watching it, in my environment).
* No GPU required (but the more CPU cores, the better).
* For scripters, there’s an option to disable keyframe reduction and export the raw motion data, so you can fine-tune it yourself

**Technical Overview:**

This uses a purely mathematical approach with no machine learning, and can make a pretty OK funscript for any source video. It’s performed well on:

* Blurry 2005 cell phone footage from that time in the walk-in.
* The stylings of directors with shaky hands, and unhealthy fondness for Dutch angles.
* The full Anime beastiary, with its wide variety of appendages.
* Good, wholesome VR with steady cameras and simple motions.
* Beach Volleyball (Don’t judge)

It doesn’t need any specific body part to be in frame, because, much like the common velociraptor, it can only see motion (though it does correct for camera movement and figure out orientation using math).

**Detailed Functionality:**

The process this application uses is:

* Compute the optical flow map between each pair of adjacent frames.
* For each optical flow pair, find the point of maximum absolute divergence (that is, most positive or negative). This provides a good extimation for the "Center of motion."
    * This is probably the most improvable part, but this worked better than PSO following the vector field, highest regional variance, and a couple other things I tried)
* Once the center is computed, project all of the optical flow vectors on the vectors between their origins the center of motion, then get the mean magnitudes.
    * This gives an approximation of how much the image is "expanding" or "contracting." 
    * The average is weighted to make sure points to the left and right of the center have the same total weight, and same for above/below. This helps to cancel out camera motion.
* Split into scenes (detect cuts based on whether the absolute magnitude of the optical flow exceeds a threshold)
* Then just integrate over time (using trapezoids), detrend/normalize, reduce to keyframes (only take points that are either a local max or min) and render to funscript. 

**Known Limitations:**

It’s good enough most of the time (I’ve been enjoying its output), but it’s no substitute for an expert scripter.

It can’t tell why whatever’s on screen is bouncing, so it scripts all motion (you may notice that there are no idle periods in the heatmap). In testing, this has been an advantage.

**Support:**

If you feel so inclined,

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N61E7FKC)
