import { useRef, useState } from "react";
import { Button, ErrorNote } from "./ui.jsx";

const MAX_MB = 10;
const ACCEPTED = ["image/jpeg", "image/png"];

/** Drop zone + thumbnail. Client-side checks are a convenience; the API validates again. */
export default function ImageDrop({ image, onFile, onPickGallery }) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState(null);
  const input = useRef(null);

  function accept(file) {
    if (!file) return;
    if (!ACCEPTED.includes(file.type)) return setError("Choose a JPEG or PNG image.");
    if (file.size > MAX_MB * 1024 * 1024) return setError(`Image is larger than ${MAX_MB} MB.`);
    setError(null);
    onFile(file);
  }

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          accept(e.dataTransfer.files?.[0]);
        }}
        className={`grid min-h-40 place-items-center overflow-hidden rounded-md border border-dashed p-2 text-center ${
          over ? "border-accent bg-accent/10" : "border-line-strong bg-surface"
        }`}
      >
        {image ? (
          <img src={image.previewUrl} alt="Selected dermoscopic image" className="max-h-52 rounded object-contain" />
        ) : (
          <p className="px-4 text-sm text-muted">
            Drop a dermoscopic image here
            <span className="num mt-1 block text-[11px] text-faint">jpeg / png · ≤ {MAX_MB} MB</span>
          </p>
        )}
      </div>
      {image && (
        <p className="num truncate text-xs text-faint" title={image.file.name}>
          {image.source === "gallery" ? "gallery · " : "upload · "}
          {image.file.name}
        </p>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Button onClick={() => input.current?.click()}>Upload image</Button>
        <Button onClick={onPickGallery}>From gallery</Button>
      </div>
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png"
        className="sr-only"
        aria-label="Upload image"
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {error && <ErrorNote>{error}</ErrorNote>}
    </div>
  );
}
