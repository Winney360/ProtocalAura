import axios from "axios";
import fs from "fs";
import FormData from "form-data";

const AI_BASE_URL = "http://127.0.0.1:8000";

export async function analyzeFrame(framePath: string) {
  const form = new FormData();
  form.append("file", fs.createReadStream(framePath));

  const response = await axios.post(
    `${AI_BASE_URL}/liveness/level2`,
    form,
    { headers: form.getHeaders() }
  );

  return response.data;
}
