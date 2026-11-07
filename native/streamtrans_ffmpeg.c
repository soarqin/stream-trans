// SPDX-License-Identifier: MIT
// Keep FFmpeg structure access behind a fixed C ABI instead of duplicating it in Rust.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>

#if LIBAVFORMAT_VERSION_MAJOR != 63 || LIBAVCODEC_VERSION_MAJOR != 63 || \
    LIBAVUTIL_VERSION_MAJOR != 61 || LIBSWRESAMPLE_VERSION_MAJOR != 7
#error streamtrans-ffmpeg requires matching FFmpeg 9 headers and libraries.
#endif

#define ST_EXPORT __declspec(dllexport)
#define ST_RATE 16000

enum { ST_OK = 0, ST_DECODE = 1, ST_TIMEOUT = 2, ST_TOO_LONG = 3, ST_INTERNAL = 4 };

typedef struct Decode {
    AVFormatContext *format;
    AVCodecContext *codec;
    AVPacket *packet;
    AVFrame *frame;
    SwrContext *resampler;
    AVChannelLayout input_layout;
    enum AVSampleFormat input_format;
    int input_rate;
    uint64_t deadline;
    uint64_t limit;
    float *samples;
    size_t count;
    size_t capacity;
    char *error;
    size_t error_capacity;
} Decode;

static int fail(Decode *d, int status, const char *message, int fferror) {
    if (d->error && d->error_capacity) {
        if (fferror < 0) {
            char detail[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(fferror, detail, sizeof(detail));
            snprintf(d->error, d->error_capacity, "%s: %s", message, detail);
        } else {
            snprintf(d->error, d->error_capacity, "%s", message);
        }
        d->error[d->error_capacity - 1] = '\0';
    }
    return status;
}

static int expired(void *opaque) {
    Decode *d = (Decode *)opaque;
    return GetTickCount64() >= d->deadline;
}

static int check_time(Decode *d) {
    return expired(d) ? fail(d, ST_TIMEOUT, "Audio decoding timed out", 0) : ST_OK;
}

static int ff_fail(Decode *d, const char *message, int code) {
    int status = check_time(d);
    if (status) return status;
    return fail(d, code == AVERROR(ENOMEM) ? ST_INTERNAL : ST_DECODE, message, code);
}

static int reserve_samples(Decode *d, size_t required) {
    size_t maximum = (size_t)d->limit + 1;
    size_t capacity;
    float *allocation;
    if (required <= d->capacity) return ST_OK;
    capacity = d->capacity ? d->capacity : 4096;
    if (capacity > maximum) capacity = maximum;
    while (capacity < required) {
        capacity = capacity > maximum / 2 ? maximum : capacity * 2;
    }
    allocation = (float *)realloc(d->samples, capacity * sizeof(float));
    if (!allocation) return fail(d, ST_INTERNAL, "Cannot allocate decoded audio", 0);
    d->samples = allocation;
    d->capacity = capacity;
    return ST_OK;
}

static int convert_samples(Decode *d, const uint8_t **input, int count, int *produced) {
    int bound, output_count, result, status;
    uint64_t remaining;
    uint8_t *output;
    *produced = 0;
    if ((status = check_time(d))) return status;
    bound = swr_get_out_samples(d->resampler, count);
    if (bound < 0) return ff_fail(d, "Cannot size resampled audio", bound);
    // One extra sample distinguishes an exact-length input from an over-limit input.
    remaining = d->limit - (uint64_t)d->count + 1;
    output_count = bound > 0 ? bound : 1;
    if ((uint64_t)output_count > remaining) output_count = (int)remaining;
    if ((status = reserve_samples(d, d->count + (size_t)output_count))) return status;
    output = (uint8_t *)(d->samples + d->count);
    result = swr_convert(d->resampler, &output, output_count, input, count);
    if ((status = check_time(d))) return status;
    if (result < 0) return ff_fail(d, "Cannot resample audio", result);
    if ((uint64_t)result > d->limit - (uint64_t)d->count) {
        return fail(d, ST_TOO_LONG, "Audio exceeds the configured sample limit", 0);
    }
    d->count += (size_t)result;
    *produced = result;
    return ST_OK;
}

static int flush_resampler(Decode *d) {
    int status, produced;
    if (!d->resampler) return ST_OK;
    do {
        status = convert_samples(d, NULL, 0, &produced);
        if (status) return status;
    } while (produced);
    return ST_OK;
}

static int process_frame(Decode *d) {
    AVFrame *frame = d->frame;
    AVChannelLayout layout = { 0 };
    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    const AVChannelLayout *source_layout;
    int rate, code, status, produced;
    if ((status = check_time(d))) return status;
    if (!frame->nb_samples) return ST_OK;
    if (frame->nb_samples < 0 || frame->format < 0) {
        return fail(d, ST_DECODE, "Decoder returned invalid audio parameters", 0);
    }
    rate = frame->sample_rate > 0 ? frame->sample_rate : d->codec->sample_rate;
    source_layout = frame->ch_layout.nb_channels > 0 ? &frame->ch_layout : &d->codec->ch_layout;
    if (rate <= 0 || source_layout->nb_channels <= 0) {
        return fail(d, ST_DECODE, "Audio has no sample rate or channel layout", 0);
    }
    if (source_layout->order == AV_CHANNEL_ORDER_UNSPEC) {
        av_channel_layout_default(&layout, source_layout->nb_channels);
    } else {
        code = av_channel_layout_copy(&layout, source_layout);
        if (code < 0) return ff_fail(d, "Cannot copy audio channel layout", code);
    }
    if (!av_channel_layout_check(&layout)) {
        av_channel_layout_uninit(&layout);
        return fail(d, ST_DECODE, "Audio has an invalid channel layout", 0);
    }
    if (!d->resampler || d->input_rate != rate || d->input_format != frame->format ||
        av_channel_layout_compare(&d->input_layout, &layout) != 0) {
        status = flush_resampler(d);
        if (status) goto done;
        swr_free(&d->resampler);
        av_channel_layout_uninit(&d->input_layout);
        code = swr_alloc_set_opts2(&d->resampler, &mono, AV_SAMPLE_FMT_FLT, ST_RATE,
                                  &layout, (enum AVSampleFormat)frame->format, rate, 0, NULL);
        if (code < 0) { status = ff_fail(d, "Cannot configure audio resampler", code); goto done; }
        // Float output otherwise keeps extra downmix gain that the previous PCM16 output normalized.
        code = av_opt_set_double(d->resampler, "rematrix_maxval", 1.0, 0);
        if (code < 0) { status = ff_fail(d, "Cannot configure audio downmix normalization", code); goto done; }
        code = swr_init(d->resampler);
        if (code < 0) { status = ff_fail(d, "Cannot initialize audio resampler", code); goto done; }
        code = av_channel_layout_copy(&d->input_layout, &layout);
        if (code < 0) { status = ff_fail(d, "Cannot retain audio channel layout", code); goto done; }
        d->input_rate = rate;
        d->input_format = (enum AVSampleFormat)frame->format;
    }
    status = convert_samples(d, (const uint8_t **)frame->extended_data, frame->nb_samples, &produced);
done:
    av_channel_layout_uninit(&layout);
    return status;
}

static int receive_frames(Decode *d, int *received, int *end) {
    int code, status;
    *received = 0;
    for (;;) {
        if ((status = check_time(d))) return status;
        code = avcodec_receive_frame(d->codec, d->frame);
        if ((status = check_time(d))) return status;
        if (code == AVERROR(EAGAIN) || code == AVERROR_EOF) {
            *end = code;
            return ST_OK;
        }
        if (code < 0) return ff_fail(d, "Cannot decode audio frame", code);
        ++*received;
        status = process_frame(d);
        av_frame_unref(d->frame);
        if (status) return status;
    }
}

static int submit_packet(Decode *d, const AVPacket *packet) {
    int code, status, received, end;
    for (;;) {
        if ((status = check_time(d))) return status;
        code = avcodec_send_packet(d->codec, packet);
        if ((status = check_time(d))) return status;
        if (code != AVERROR(EAGAIN)) break;
        status = receive_frames(d, &received, &end);
        if (status) return status;
        if (!received) return fail(d, ST_INTERNAL, "Audio decoder made no progress", 0);
    }
    if (code == AVERROR_EOF && !packet) return ST_OK;
    if (code < 0) return ff_fail(d, "Cannot submit audio packet", code);
    status = receive_frames(d, &received, &end);
    if (status) return status;
    if (!packet && end != AVERROR_EOF) {
        return fail(d, ST_INTERNAL, "Audio decoder did not finish draining", 0);
    }
    return ST_OK;
}

ST_EXPORT unsigned st_audio_abi_version(void) { return 1; }

ST_EXPORT void st_audio_versions(unsigned out[4]) {
    if (!out) return;
    out[0] = avformat_version();
    out[1] = avcodec_version();
    out[2] = avutil_version();
    out[3] = swresample_version();
}

ST_EXPORT void st_audio_free(float *samples) { free(samples); }

ST_EXPORT int st_audio_decode(const char *utf8_path, uint64_t max_samples,
                             uint32_t timeout_ms, float **samples, size_t *sample_count,
                             char *error, size_t error_capacity) {
    Decode d = { 0 };
    AVDictionary *options = NULL;
    const AVCodec *decoder = NULL;
    int stream = -1, code, status = ST_OK;
    unsigned i;
    d.error = error;
    d.error_capacity = error_capacity;
    d.deadline = GetTickCount64() + timeout_ms;
    d.limit = max_samples;
    if (error && error_capacity) error[0] = '\0';
    if (samples) *samples = NULL;
    if (sample_count) *sample_count = 0;
    if (!utf8_path || !samples || !sample_count || max_samples >= SIZE_MAX / sizeof(float)) {
        return fail(&d, ST_INTERNAL, "Invalid audio decoder arguments", 0);
    }
    if ((avformat_version() >> 16) != 63 || (avcodec_version() >> 16) != 63 ||
        (avutil_version() >> 16) != 61 || (swresample_version() >> 16) != 7) {
        return fail(&d, ST_INTERNAL, "FFmpeg DLL versions do not match this bridge", 0);
    }
    if ((status = check_time(&d))) return status;
    d.format = avformat_alloc_context();
    if (!d.format) return fail(&d, ST_INTERNAL, "Cannot allocate audio demuxer", 0);
    d.format->interrupt_callback.callback = expired;
    d.format->interrupt_callback.opaque = &d;
    code = av_dict_set(&options, "protocol_whitelist", "file,pipe", 0);
    if (code < 0) { status = ff_fail(&d, "Cannot configure audio protocols", code); goto done; }
    code = avformat_open_input(&d.format, utf8_path, NULL, &options);
    av_dict_free(&options);
    if (code < 0) { status = ff_fail(&d, "Cannot open audio", code); goto done; }
    if ((status = check_time(&d))) goto done;
    code = avformat_find_stream_info(d.format, NULL);
    if (code < 0) { status = ff_fail(&d, "Cannot read audio stream information", code); goto done; }
    if ((status = check_time(&d))) goto done;
    // Match the previous -map 0:a:0 behavior, including an unsupported first audio track.
    for (i = 0; i < d.format->nb_streams; ++i) {
        if (d.format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            stream = (int)i;
            decoder = avcodec_find_decoder(d.format->streams[i]->codecpar->codec_id);
            break;
        }
    }
    if (stream < 0 || !decoder) {
        status = fail(&d, ST_DECODE, "No supported first audio stream", 0);
        goto done;
    }
    d.codec = avcodec_alloc_context3(decoder);
    if (!d.codec) { status = fail(&d, ST_INTERNAL, "Cannot allocate audio decoder", 0); goto done; }
    code = avcodec_parameters_to_context(d.codec, d.format->streams[stream]->codecpar);
    if (code < 0) { status = ff_fail(&d, "Cannot copy audio stream parameters", code); goto done; }
    d.codec->pkt_timebase = d.format->streams[stream]->time_base;
    d.codec->thread_count = 1;
    code = avcodec_open2(d.codec, decoder, NULL);
    if (code < 0) { status = ff_fail(&d, "Cannot initialize audio decoder", code); goto done; }
    d.packet = av_packet_alloc();
    d.frame = av_frame_alloc();
    if (!d.packet || !d.frame) { status = fail(&d, ST_INTERNAL, "Cannot allocate audio frame", 0); goto done; }
    for (;;) {
        if ((status = check_time(&d))) goto done;
        code = av_read_frame(d.format, d.packet);
        if ((status = check_time(&d))) goto done;
        if (code == AVERROR_EOF) break;
        if (code == AVERROR(EAGAIN)) continue;
        if (code < 0) { status = ff_fail(&d, "Cannot read audio packet", code); goto done; }
        if (d.packet->stream_index == stream) status = submit_packet(&d, d.packet);
        av_packet_unref(d.packet);
        if (status) goto done;
    }
    status = submit_packet(&d, NULL);
    if (!status) status = flush_resampler(&d);
    if (!status) status = check_time(&d);
done:
    av_dict_free(&options);
    av_packet_free(&d.packet);
    av_frame_free(&d.frame);
    swr_free(&d.resampler);
    av_channel_layout_uninit(&d.input_layout);
    avcodec_free_context(&d.codec);
    avformat_close_input(&d.format);
    if (!status) {
        *samples = d.samples;
        *sample_count = d.count;
    } else {
        free(d.samples);
    }
    return status;
}
