import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";
import path from "path";
import fs from "fs";

ffmpeg.setFfmpegPath(ffmpegPath as string);

export const extractFrames = (
  videoPath: string,
  outputDir: string
): Promise<string[]> => {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    ffmpeg(videoPath)
      .outputOptions([
        "-vf fps=1", // 1 frame per second
      ])
      .output(path.join(outputDir, "frame-%03d.jpg"))
      .on("end", () => {
        const frames = fs
          .readdirSync(outputDir)
          .map((f) => path.join(outputDir, f));
        resolve(frames);
      })
      .on("error", reject)
      .run();
  });
};
