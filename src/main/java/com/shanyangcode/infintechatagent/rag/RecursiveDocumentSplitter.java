package com.shanyangcode.infintechatagent.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
public class RecursiveDocumentSplitter implements DocumentSplitter {

    private final int maxChunkSize;
    private final int chunkOverlap;
    private final List<String> separators;

    public RecursiveDocumentSplitter(int maxChunkSize, int chunkOverlap) {
        this.maxChunkSize = maxChunkSize;
        this.chunkOverlap = chunkOverlap;
        // 中文优化：先按中文标点，再按段落、换行、空格
        this.separators = Arrays.asList("。", "！", "？", "\n\n", "\n", " ", "");
    }

    public RecursiveDocumentSplitter(int maxChunkSize, int chunkOverlap, List<String> separators) {
        this.maxChunkSize = maxChunkSize;
        this.chunkOverlap = chunkOverlap;
        this.separators = separators;
    }

    @Override
    public List<TextSegment> split(Document document) {
        log.info("📄 开始递归分割文档 - 原文长度:{}字符", document.text().length());
        return split(document.text(), document.metadata());
    }

    public List<TextSegment> split(String text, Metadata metadata) {
        List<String> chunks = splitText(text, 0);
        List<TextSegment> segments = new ArrayList<>();

        for (String chunk : chunks) {
            if (!chunk.trim().isEmpty()) {
                segments.add(TextSegment.from(chunk, metadata));
            }
        }

        log.info("✅ 递归分割完成: {}字符 → {}个片段 (平均{}字符/片段)",
            text.length(), segments.size(), text.length() / Math.max(1, segments.size()));
        return segments;
    }

    private List<String> splitText(String text, int separatorIndex) {
        if (text.length() <= maxChunkSize) {
            return Arrays.asList(text);
        }

        if (separatorIndex >= separators.size()) {
            return splitByCharacter(text);
        }

        String separator = separators.get(separatorIndex);
        List<String> result = new ArrayList<>();

        if (separator.isEmpty()) {
            return splitByCharacter(text);
        }

        String[] parts = text.split(separator, -1);
        StringBuilder currentChunk = new StringBuilder();

        for (int i = 0; i < parts.length; i++) {
            String part = parts[i];

            if (part.length() > maxChunkSize) {
                if (currentChunk.length() > 0) {
                    result.add(currentChunk.toString());
                    currentChunk = new StringBuilder();
                }
                result.addAll(splitText(part, separatorIndex + 1));
            } else {
                String testChunk = currentChunk.length() == 0 ? part :
                    currentChunk.toString() + separator + part;

                if (testChunk.length() <= maxChunkSize) {
                    currentChunk = new StringBuilder(testChunk);
                } else {
                    if (currentChunk.length() > 0) {
                        result.add(currentChunk.toString());
                    }
                    currentChunk = new StringBuilder(part);
                }
            }
        }

        if (currentChunk.length() > 0) {
            result.add(currentChunk.toString());
        }

        return addOverlap(result);
    }

    private List<String> splitByCharacter(String text) {
        List<String> result = new ArrayList<>();
        for (int i = 0; i < text.length(); i += maxChunkSize) {
            result.add(text.substring(i, Math.min(i + maxChunkSize, text.length())));
        }
        return result;
    }

    private List<String> addOverlap(List<String> chunks) {
        if (chunkOverlap == 0 || chunks.size() <= 1) {
            return chunks;
        }

        List<String> result = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);

            if (i > 0 && chunkOverlap > 0) {
                String prevChunk = chunks.get(i - 1);
                int overlapStart = Math.max(0, prevChunk.length() - chunkOverlap);
                String overlap = prevChunk.substring(overlapStart);
                chunk = overlap + chunk;
            }

            result.add(chunk);
        }
        return result;
    }
}
