package com.shanyangcode.infintechatagent.agent;

/**
 * Agent基础接口
 */
public interface Agent {

    /**
     * 执行Agent任务
     * @param sessionId 会话ID
     * @param input 输入内容
     * @return 执行结果
     */
    String execute(Long sessionId, String input);

    /**
     * 获取Agent名称
     * @return Agent名称
     */
    String getAgentName();
}
