# 🎙️ Whisper Audio Splitter

A powerful tool that automatically splits audio files when specific words or patterns are detected using speech-to-text technology. Perfect for breaking down long recordings into meaningful segments!

The original idea was simple: I wanted to create Anki flashcards by recording long sessions of me speaking cards out loud, saying "STOP" between each card. This tool then uses speech recognition to detect these stops and split the recording into individual card segments, preserving the original audio for reference. That worked very well for me so I ended up making it usable for others!

## 🌟 Key Features

- 🔍 Splits audio when specific words are detected (default: "stop")
- 🎯 Multiple STT backends (OpenAI Whisper via Replicate API, Deepgram API)
- 🧹 Smart silence removal and audio preprocessing
- 📦 Batch processing support
- 💾 Caching of transcription results
- ⚡ Parallel processing for long audio files

## 🚀 Quick Start

1. Set up your API key:
```bash
export REPLICATE_API_KEY="your-key-here"
# or
export DEEPGRAM_API_KEY="your-key-here"
```

2. Basic usage:
```bash
# Split files with default settings (French, "stop" as split word)
python -m whisper_audio_splitter \
    --untouched_dir=/path/to/input \
    --splitted_dir=/path/to/output \
    --done_dir=/path/to/processed

# Customize language and split words
python -m whisper_audio_splitter \
    --language=en \
    --stop_list="['stop','pause','break']" \
    --untouched_dir=./input \
    --splitted_dir=./output \
    --done_dir=./done
```

## 📋 Requirements

- Python 3.11+
- REPLICATE_API_KEY or DEEPGRAM_API_KEY environment variable
- Input/output directories must exist
- Audio files in mp3 or wav format

## 🔧 How It Works

1. Load audio files from `untouched_dir`
2. Remove silence (multiple methods supported)
3. Transcribe the text using STT (using either either Deepgram, Replicate or even self hosted whisper)
4. Find the time location of where the prompt is pronounced (by default 'Stop' but can be any string)
5. Split the original unsilenced audio into segments at those location and put the results in `splitted_dir`
6. Moves the large files (original and unsilenced) to `done_dir`

## 📝 Note on Project Status

This tool was originally developed as part of [Voice2Anki](https://github.com/thiswillbeyourgithub/Voice2Anki), my project for creating Anki flashcards from voice recordings. Since moving it to a standalone project, I haven't needed to use it extensively, so there might be minor issues. However, don't hesitate to open an issue if you encounter any problems - they're probably quick fixes!

If you're interested in Anki-related projects, check out my other repositories - I have several tools for enhancing the Anki experience!

## 🤝 Contributing

Issues and pull requests are welcome! The codebase is well-documented and should be easy to understand and modify.
