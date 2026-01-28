// File: server/src/utils/extractAudio.ts (NEW FILE)
import ffmpeg from "fluent-ffmpeg";
import fs from "fs";
import path from "path";

export const extractAudio = (
  videoPath: string,
  outputDir: string
): Promise<string> => {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const audioPath = path.join(outputDir, "audio.wav");

    ffmpeg(videoPath)
      .outputOptions([
        "-vn", // No video
        "-acodec", "pcm_s16le", // WAV format
        "-ar", "22050", // Sample rate
        "-ac", "1" // Mono
      ])
      .output(audioPath)
      .on("end", () => {
        if (fs.existsSync(audioPath)) {
          resolve(audioPath);
        } else {
          reject(new Error("Audio extraction failed"));
        }
      })
      .on("error", reject)
      .run();
  });
};

export const hasAudioStream = (videoPath: string): Promise<boolean> => {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
        reject(err);
        return;
      }
      
      const hasAudio = metadata.streams.some(
        (stream: any) => stream.codec_type === "audio"
      );
      resolve(hasAudio);
    });
  });
};