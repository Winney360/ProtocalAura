// File: server/src/utils/extractAudio.ts
import ffmpeg from "fluent-ffmpeg";
import fs from "fs";
import path from "path";
import * as dotenv from 'dotenv'; // Add this import

// Load environment variables
dotenv.config();

const ffmpegPath = process.env.FFMPEG_PATH;
const ffprobePath = process.env.FFPROBE_PATH;

if (ffmpegPath) {
  ffmpeg.setFfmpegPath(ffmpegPath);
  console.log(`FFmpeg path configured: ${ffmpegPath}`);
} else {
  console.warn('FFMPEG_PATH not set in environment variables');
}

if (ffprobePath) {
  ffmpeg.setFfprobePath(ffprobePath);
  console.log(`FFprobe path configured: ${ffprobePath}`);
} else {
  console.warn('FFPROBE_PATH not set in environment variables');
}

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
      .on("error", (err) => {
        console.error(`Audio extraction error: ${err.message}`);
        reject(err);
      })
      .run();
  });
};

export const hasAudioStream = (videoPath: string): Promise<boolean> => {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
        console.error(`FFprobe error: ${err.message}`);
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