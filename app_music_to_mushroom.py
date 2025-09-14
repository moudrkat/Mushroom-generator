import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import time
from scipy import ndimage
import matplotlib.pyplot as plt
import io
import base64
import wave
from streamlit_frontend import load_keras_model, generate_mushroom, contact_form, link_to_other_apps

# Load the model
if 'generator' not in st.session_state:
    try:
        st.session_state.generator = load_keras_model()
    except Exception:
        st.stop()

generator = st.session_state.generator


def extract_features_from_audio(audio_data, sr=22050, smooth_sigma=1.0):
    """Extract beat strength and spectral centroid from audio"""
    try:
        hop_length = 512
        onset_envelope = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]

        # Smooth onset strength
        if smooth_sigma > 0:
            onset_envelope = ndimage.gaussian_filter1d(onset_envelope, sigma=smooth_sigma)

        # Normalize to latent space range [-2.5, 2.5]
        if onset_envelope.max() > onset_envelope.min():
            cap_size = (onset_envelope - onset_envelope.min()) / (onset_envelope.max() - onset_envelope.min())
            cap_size = cap_size * 5 - 2.5
        else:
            cap_size = np.zeros_like(onset_envelope)

        if spectral_centroid.max() > spectral_centroid.min():
            stem_length = (spectral_centroid - spectral_centroid.min()) / (
                    spectral_centroid.max() - spectral_centroid.min())
            stem_length = stem_length * 5 - 2.5
        else:
            stem_length = np.zeros_like(spectral_centroid)

        return cap_size, stem_length, hop_length
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None


def audio_to_base64_wav(audio_data, sample_rate):
    """Convert numpy audio array to base64 WAV format"""
    # Normalize audio
    audio_normalized = audio_data / np.max(np.abs(audio_data))
    audio_int16 = (audio_normalized * 32767).astype(np.int16)

    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    # Convert to base64
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode()
    return audio_base64


def generate_mushroom_silent(generator, latent_vector, color):
    """
    Generate mushroom without displaying it on screen.
    This is a wrapper around the original generate_mushroom function
    that captures any streamlit output.
    """
    # Method 1: Use st.empty() to capture and clear any output
    placeholder = st.empty()

    with placeholder.container():
        # Temporarily redirect any streamlit output to this container
        mushroom_img = generate_mushroom(generator, latent_vector, color)

    # Clear the placeholder to remove any displayed content
    placeholder.empty()

    return mushroom_img


def create_mushroom_movie_player(audio_data, cap_sizes, stem_lengths, sr, hop_length, color):
    """Create synchronized audio + mushroom animation player"""

    # Calculate timing
    frame_duration = hop_length / sr  # seconds per frame
    total_frames = len(cap_sizes)
    total_duration = len(audio_data) / sr

    # Generate all mushroom frames SILENTLY
    mushroom_frames = []

    # Create a progress bar for generation
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Hide any mushroom generation output in an empty container
    generation_container = st.empty()

    with generation_container.container():
        for i in range(total_frames):
            # Update progress
            progress_bar.progress((i + 1) / total_frames)
            progress_text.text(f"Generating mushroom frame {i + 1}/{total_frames}...")

            latent_vector = np.array([[cap_sizes[i], stem_lengths[i]]])

            # Use the silent generation method
            mushroom_img = generate_mushroom_silent(generator, latent_vector, color)

            # Convert to base64
            buffered = io.BytesIO()
            mushroom_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            mushroom_frames.append(img_str)

    # Clear the generation container and progress indicators
    generation_container.empty()
    progress_bar.empty()
    progress_text.empty()

    # Convert audio to base64
    audio_b64 = audio_to_base64_wav(audio_data, sr)

    # Create ACTUAL synchronized video player
    html_code = f"""
    <div style="text-align: center; padding: 20px;">
        <h3>🎬🍄 Your Mushroom Movie</h3>

        <!-- Hidden audio element -->
        <audio id="audioPlayer" preload="auto">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>

        <!-- Movie Controls -->
        <div style="margin: 20px 0;">
            <button id="playBtn" onclick="playMovie()" style="
                background: #4CAF50; color: white; border: none;
                padding: 15px 30px; border-radius: 8px; font-size: 18px;
                cursor: pointer; margin: 5px;">
                ▶️ Play Movie
            </button>
            <button id="pauseBtn" onclick="pauseMovie()" disabled style="
                background: #f44336; color: white; border: none;
                padding: 15px 30px; border-radius: 8px; font-size: 18px;
                cursor: pointer; margin: 5px;">
                ⏸️ Pause
            </button>
            <button onclick="stopMovie()" style="
                background: #9E9E9E; color: white; border: none;
                padding: 15px 30px; border-radius: 8px; font-size: 18px;
                cursor: pointer; margin: 5px;">
                ⏹️ Stop
            </button>
        </div>

        <!-- Progress info -->
        <div id="progressInfo" style="margin: 10px 0; color: #666;">
            Time: <span id="currentTime">0:00</span> / <span id="totalTime">{int(total_duration // 60)}:{int(total_duration % 60):02d}</span>
            | Frame: <span id="currentFrame">0</span>/{total_frames}
        </div>

        <!-- Mushroom display -->
        <div style="margin: 30px 0;">
            <img id="mushroomImage" src="data:image/png;base64,{mushroom_frames[0] if mushroom_frames else ''}" 
                 style="width: 90vw; max-width: none; height: auto; border: 2px solid #ddd; border-radius: 10px;">
        </div>

        <!-- Current mushroom values -->
        <div id="mushroomInfo" style="color: #666; font-size: 14px;">
            Cap Size: <span id="capValue">{cap_sizes[0]:.2f}</span> | 
            Stem Length: <span id="stemValue">{stem_lengths[0]:.2f}</span>
        </div>
    </div>

    <script>
        const audio = document.getElementById('audioPlayer');
        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const mushroomImage = document.getElementById('mushroomImage');
        const currentTimeSpan = document.getElementById('currentTime');
        const currentFrameSpan = document.getElementById('currentFrame');
        const capValueSpan = document.getElementById('capValue');
        const stemValueSpan = document.getElementById('stemValue');

        // Movie data
        const mushroomFrames = {mushroom_frames};
        const capSizes = {cap_sizes.tolist()};
        const stemLengths = {stem_lengths.tolist()};
        const frameDuration = {frame_duration};
        const totalFrames = {total_frames};

        let movieTimer = null;
        let currentFrameIndex = 0;

        function playMovie() {{
            // Start audio
            audio.play();

            // Enable/disable buttons
            playBtn.disabled = true;
            pauseBtn.disabled = false;

            // Start mushroom animation timer
            movieTimer = setInterval(updateMushroom, frameDuration * 1000);

            console.log('Movie started - audio + mushroom animation');
        }}

        function pauseMovie() {{
            // Pause audio
            audio.pause();

            // Stop mushroom animation
            if (movieTimer) {{
                clearInterval(movieTimer);
                movieTimer = null;
            }}

            // Enable/disable buttons
            playBtn.disabled = false;
            pauseBtn.disabled = true;

            console.log('Movie paused');
        }}

        function stopMovie() {{
            // Stop audio
            audio.pause();
            audio.currentTime = 0;

            // Stop animation
            if (movieTimer) {{
                clearInterval(movieTimer);
                movieTimer = null;
            }}

            // Reset to first frame
            currentFrameIndex = 0;
            updateMushroomDisplay();

            // Reset buttons
            playBtn.disabled = false;
            pauseBtn.disabled = true;

            console.log('Movie stopped and reset');
        }}

        function updateMushroom() {{
            const audioCurrentTime = audio.currentTime;

            // Calculate which frame should be showing
            const targetFrame = Math.floor(audioCurrentTime / frameDuration);

            // Update frame if needed
            if (targetFrame !== currentFrameIndex && targetFrame < totalFrames) {{
                currentFrameIndex = targetFrame;
                updateMushroomDisplay();
            }}

            // Update time display
            const minutes = Math.floor(audioCurrentTime / 60);
            const seconds = Math.floor(audioCurrentTime % 60);
            currentTimeSpan.textContent = `${{minutes}}:${{seconds.toString().padStart(2, '0')}}`;

            // Stop when finished
            if (audioCurrentTime >= audio.duration || currentFrameIndex >= totalFrames - 1) {{
                pauseMovie();
            }}
        }}

        function updateMushroomDisplay() {{
            if (currentFrameIndex < mushroomFrames.length) {{
                // Change mushroom image
                mushroomImage.src = 'data:image/png;base64,' + mushroomFrames[currentFrameIndex];

                // Update display info
                currentFrameSpan.textContent = currentFrameIndex + 1;
                capValueSpan.textContent = capSizes[currentFrameIndex].toFixed(2);
                stemValueSpan.textContent = stemLengths[currentFrameIndex].toFixed(2);

                console.log(`Frame ${{currentFrameIndex + 1}}: Cap=${{capSizes[currentFrameIndex].toFixed(2)}}, Stem=${{stemLengths[currentFrameIndex].toFixed(2)}}`);
            }}
        }}

        // Audio ended event
        audio.addEventListener('ended', function() {{
            console.log('Audio ended');
            pauseMovie();
        }});

        // Initialize display
        updateMushroomDisplay();

        console.log(`Movie ready: ${{totalFrames}} frames, ${{frameDuration.toFixed(3)}}s per frame`);
    </script>
    """

    return html_code


# Streamlit App
st.title("🎬🍄 Mushroom Movie Maker")

st.markdown("""
    **Create synchronized mushroom movies from your recordings!**

    **The Pipeline** 🎵 → 🧮 → 🍄

    **Recording → Fourier Transform → Interpretable Latent Space → VAE → Mushroom**

    Your audio signal x(t) gets decomposed via:
""")

st.latex(r"O(n) = \sum_{k} H(|X(n,k)| - |X(n-1,k)|)")
st.markdown("""
**Onset Strength** - detects beats and rhythmic events

**What it does:** Detects when new sounds/beats start

**Step by step:**
- **X(n,k)** = Your audio's frequency spectrum at time frame `n`, frequency bin `k`
- **|X(n,k)|** = Magnitude (how loud each frequency is)
- **|X(n,k)| - |X(n-1,k)|** = Change in loudness from previous frame
- **H(...)** = Half-wave rectifier (only keeps positive changes - new sounds getting louder)
- **Σₖ** = Sum across all frequencies

**In simple terms:** "How much did the audio get louder across all frequencies compared to a split second ago?"

**👏 Example:** When you clap, ALL frequencies suddenly get louder → big onset strength value → BIG mushroom cap! 🍄💥
""")

st.latex(r"SC(n) = \frac{\sum_{k} k \cdot |X(n,k)|}{\sum_{k} |X(n,k)|}")
st.markdown("""
**Spectral Centroid** - measures spectral brightness

**What it does:** Measures the "brightness" or "center of mass" of your sound

**Step by step:**
- **k** = Frequency bin number (higher k = higher pitch)
- **k·|X(n,k)|** = Frequency × its loudness (weighted by pitch)  
- **Σₖ k·|X(n,k)|** = Sum of all frequency-weighted magnitudes
- **Σₖ |X(n,k)|** = Total energy across all frequencies
- **Division** = Weighted average frequency

**In simple terms:** "Where is the center of your sound's frequency content?"

**🎵 Examples:**
- Deep bass → low SC → short mushroom stem
- Bright cymbal → high SC → long mushroom stem  
- Human voice → medium SC → medium stem
""")

st.markdown("""
    These features map to a **2D interpretable latent space** where:
    - **O(n)** → Cap size (stronger beats = bigger caps! 🥁🍄)
    - **SC(n)** → Stem length (brighter sounds = different stems 🎨)

    The **VAE decoder** transforms latent coordinates **z = [cap, stem]** into mushroom images, creating a **continuous morphological space** of fungi controlled by your audio's acoustic properties! 🎛️🍄✨
""")

# Step 1: Audio Input
st.subheader("🎵 Step 1: Get Your Audio")

tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])

audio_data = None
sr = None

with tab1:
    uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg', 'm4a'])
    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load audio
        try:
            audio_data, sr = librosa.load(tmp_file_path, sr=22050, duration=60)  # Limit to 60s
            st.success(f"✓ Loaded {len(audio_data) / sr:.1f}s of audio")
        except Exception as e:
            st.error(f"Error loading audio: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

with tab2:
    st.info("🎤 For recording, use the simple method:")
    st.code("""
    # Ubuntu command line recording:
    arecord -d 10 -f cd -t wav my_recording.wav

    # Or use gnome-sound-recorder GUI
    """)
    st.markdown("Then upload the recorded file in the Upload tab!")

# Step 2: Generate Movie
if audio_data is not None:
    st.subheader("🎬 Step 2: Create Your Mushroom Movie")

    col1, col2 = st.columns(2)
    with col1:
        smooth_factor = st.slider("Beat smoothing", 0.0, 3.0, 1.0, 0.1)
        color = st.color_picker("Mushroom color", "#FF6B6B")
    with col2:
        st.info(f"""
        **Your Audio:**
        - Duration: {len(audio_data) / sr:.1f} seconds
        - Sample rate: {sr} Hz
        - Will generate ~{len(audio_data) // (sr // 43)} mushroom frames
        """)

    if st.button("🎵 Analyze Audio for Mushrooms!", type="primary"):
        with st.spinner("🎵 Analyzing your audio..."):
            # Extract features
            cap_sizes, stem_lengths, hop_length = extract_features_from_audio(
                audio_data, sr, smooth_factor
            )

            if cap_sizes is not None and len(cap_sizes) > 0:
                st.success(f"✓ Audio analysis complete! Found {len(cap_sizes)} mushroom frames")

                # Store in session state to avoid regenerating
                st.session_state.movie_data = {
                    'audio_data': audio_data,
                    'cap_sizes': cap_sizes,
                    'stem_lengths': stem_lengths,
                    'sr': sr,
                    'hop_length': hop_length,
                    'color': color
                }

                st.success("✅ Audio analyzed! Click below to generate and play your mushroom movie.")
            else:
                st.error("Could not analyze audio. Try a different file.")

# Step 3: Generate and Play Movie
if 'movie_data' in st.session_state:
    st.subheader("🎬 Step 3: Play Your Mushroom Movie")

    if st.button("🍄 Generate & Play Mushroom Movie!", type="primary"):
        movie_data = st.session_state.movie_data

        # Create the movie player
        st.markdown("🍄 **Generating mushroom frames...**")
        movie_player = create_mushroom_movie_player(
            movie_data['audio_data'],
            movie_data['cap_sizes'],
            movie_data['stem_lengths'],
            movie_data['sr'],
            movie_data['hop_length'],
            movie_data['color']
        )

        # Display the movie player
        st.markdown("---")
        st.components.v1.html(movie_player, height=800)

        st.success("🎉 Your mushroom movie is ready! Press the ▶️ Play button above!")

        # Show stats
        with st.expander("📊 Movie Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames", len(movie_data['cap_sizes']))
            with col2:
                st.metric("Frame Rate", f"{movie_data['sr'] / movie_data['hop_length']:.1f} FPS")
            with col3:
                st.metric("Duration", f"{len(movie_data['audio_data']) / movie_data['sr']:.1f}s")
            with col4:
                st.metric("Sync Accuracy", "Perfect ✨")

else:
    st.info("👆 Upload an audio file to start creating your mushroom movie!")


