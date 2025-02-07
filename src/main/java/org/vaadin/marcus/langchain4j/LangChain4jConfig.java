package org.vaadin.marcus.langchain4j;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.http.client.HttpClientBuilder;
import dev.langchain4j.http.client.spring.restclient.SpringRestClientBuilder;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
import dev.langchain4j.model.Tokenizer;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.HuggingFaceTokenizer;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;
import java.util.Collections;

@Configuration
public class LangChain4jConfig {

    @Bean
    Tokenizer tokenizer() {
        return new HuggingFaceTokenizer();
    }

    @Bean
    ChatMemoryProvider chatMemoryProvider(Tokenizer tokenizer) {
        return chatId -> TokenWindowChatMemory.withMaxTokens(1000, tokenizer);
    }

    @Bean
    EmbeddingStore<TextSegment> embeddingStore() {
        return new InMemoryEmbeddingStore<>();
    }

    @Bean
    EmbeddingModel embeddingModel(
            @Value("${langchain4j.ollama.streaming-chat-model.base-url}") String ollamaBaseUrl,
            @Value("${langchain4j.ollama.streaming-chat-model.model-name}") String modelName
    ) {
        return new OllamaEmbeddingModel(
                new SpringRestClientBuilder(),
                ollamaBaseUrl,
                modelName,
                Duration.ofMinutes(1),
                5,
                true,
                true,
                Collections.emptyMap()
        );
    }


    @Bean
    ContentRetriever contentRetriever(
            EmbeddingStore<TextSegment> embeddingStore,
            EmbeddingModel embeddingModel
    ) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();
    }
}
