package com.shanyangcode.infintechatagent;

import dev.langchain4j.community.store.embedding.redis.spring.RedisEmbeddingStoreAutoConfiguration;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;


@SpringBootApplication(exclude = {RedisEmbeddingStoreAutoConfiguration.class})
@EnableScheduling
public class InfiniteChatAgentApplication {

    public static void main(String[] args) {
        SpringApplication.run(InfiniteChatAgentApplication.class, args);
    }

}
