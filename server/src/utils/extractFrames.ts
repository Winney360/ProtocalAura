// File: server/src/utils/extractFrames.ts (UPDATED)
import ffmpeg from "fluent-ffmpeg";
import fs from "fs";
import path from "path";

export const extractFrames = (
  videoPath: string,
  outputDir: string,
  frameCount: number = 30
): Promise<string[]> => {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // First get video duration
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
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
            
            resolve(frameFiles);
          });
        })
        .on("error", reject)
        .run();
    });
  });
};