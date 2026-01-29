import ffmpeg from "fluent-ffmpeg";
import fs from "fs";
import path from "path";
import * as dotenv from 'dotenv'; 

// Load environment variables
dotenv.config();

const ffmpegPath = process.env.FFMPEG_PATH;
const ffprobePath = process.env.FFPROBE_PATH;

if (ffmpegPath) {
  ffmpeg.setFfmpegPath(ffmpegPath);
  console.log(`[extractFrames] FFmpeg path configured: ${ffmpegPath}`);
} else {
  console.warn('[extractFrames] FFMPEG_PATH not set in environment variables');
}

if (ffprobePath) {
  ffmpeg.setFfprobePath(ffprobePath);
  console.log(`[extractFrames] FFprobe path configured: ${ffprobePath}`);
} else {
  console.warn('[extractFrames] FFPROBE_PATH not set in environment variables');
}

export const extractFrames = (
  videoPath: string,
  outputDir: string,
  frameCount: number = 30
): Promise<string[]> => {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // First get video duration using ffprobe
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
        console.error(`[extractFrames] FFprobe error: ${err.message}`);
        reject(err);
        return;
      }

      const duration = metadata.format.duration || 0;
      
      // Calculate interval to extract specified number of frames
      const interval = duration / frameCount;
      
      const frames: string[] = [];
      let frameIndex = 0;

      ffmpeg(videoPath)
        .outputOptions([
          `-vf`, `fps=1/${interval}`,  // Extract at calculated intervals
          `-q:v`, `2`,                  // Quality
          `-vsync`, `vfr`               // Variable frame rate
        ])
        .output(path.join(outputDir, "frame_%03d.jpg"))
        .on("start", (commandLine) => {
          console.log(`[extractFrames] FFmpeg command: ${commandLine}`);
        })
        .on("end", () => {
          // Read all extracted frames
          fs.readdir(outputDir, (err, files) => {
            if (err) {
              reject(err);
              return;
            }
            
            // Filter for jpg files and sort them
            const frameFiles = files
              .filter(file => file.endsWith(".jpg"))
              .sort((a, b) => {
                const numA = parseInt(a.match(/_(\d+)/)?.[1] || "0");
                const numB = parseInt(b.match(/_(\d+)/)?.[1] || "0");
                return numA - numB;
              })
              .slice(0, frameCount) // Ensure we don't exceed requested count
              .map(file => path.join(outputDir, file));
            
            console.log(`[extractFrames] Extracted ${frameFiles.length} frames`);
            resolve(frameFiles);
          });
        })
        .on("error", (err) => {
          console.error(`[extractFrames] FFmpeg error: ${err.message}`);
          reject(err);
        })
        .run();
    });
  });
};