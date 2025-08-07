from typing import Dict, Any
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.app.core.config import settings
from .state import AgentState, WorkflowStep
from .cache_service import SemanticCacheService
import logging

logger = logging.getLogger(__name__)

class ResponseAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1
        )
        self.cache_service = SemanticCacheService()
    
    async def process(self, state: AgentState) -> AgentState:
        """Generate response using retrieved context"""
        try:
            # If we have cached response, personalize it
            if state.get("cached_response"):
                response, confidence = await self._personalize_cached_response(
                    state["cached_response"], 
                    state["query"]
                )
                state["response"] = response
                state["confidence_score"] = confidence
            else:
                # Generate new response from retrieved context
                response, confidence = await self._generate_response(
                    state["query"], 
                    state.get("retrieved_context", []),
                    state["category"]
                )
                state["response"] = response
                state["confidence_score"] = confidence
            
            # Determine next step based on confidence
            if state["confidence_score"] >= 0.85:
                # High confidence - resolve ticket
                state["current_step"] = WorkflowStep.COMPLETION.value
                
                # Cache the response for future use
                try:
                    await self.cache_service.store_response(
                        query=state["query"],
                        response=state["response"],
                        confidence=state["confidence_score"],
                        category=state["category"]
                    )
                except Exception as e:
                    logger.error(f"Failed to cache response: {str(e)}")
                    
            else:
                # Medium/Low confidence - escalate to human
                state["requires_escalation"] = True
                state["current_step"] = WorkflowStep.ESCALATION.value
            
            return state
            
        except Exception as e:
            logger.error(f"Error in response agent: {str(e)}")
            state["error_message"] = str(e)
            state["requires_escalation"] = True
            state["current_step"] = WorkflowStep.ESCALATION.value
            return state
    
    async def _personalize_cached_response(self, cached_response: str, query: str) -> tuple[str, float]:
        """Personalize a cached response for the current query"""
        try:
            personalization_prompt = f"""
            You have a cached response that was helpful for a similar query. Please personalize it for the current specific query.
            
            Current Query: {query}
            
            Cached Response: {cached_response}
            
            Please:
            1. Adapt the response to directly address the current query
            2. Keep the helpful information from the cached response
            3. Make it feel natural and personalized
            4. Maintain a helpful, professional tone
            
            Provide your response in JSON format:
            {{
                "personalized_response": "your personalized response here",
                "confidence_score": 0.XX (between 0.0 and 1.0)
            }}
            """
            
            response = await self.llm.ainvoke(personalization_prompt)
            
            try:
                result = json.loads(response.content)
                return result.get("personalized_response", cached_response), result.get("confidence_score", 0.8)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in personalization: {str(e)}")
                logger.debug(f"Raw response: {response.content}")
                # Fallback to original cached response
                return cached_response, 0.8
            
        except Exception as e:
            logger.error(f"Personalization error: {str(e)}")
            # Fallback to original cached response
            return cached_response, 0.8
    
    async def _generate_response(self, query: str, context_chunks: list, category: str) -> tuple[str, float]:
        """Generate new response from retrieved context"""
        try:
            # Prepare context
            context_text = ""
            if context_chunks:
                context_text = "\n\n".join([
                    f"Source: {chunk['filename']}\nContent: {chunk['content'][:800]}"
                    for chunk in context_chunks
                ])
            
            response_prompt = f"""
            You are a helpful support agent for Masai School LMS. A student has submitted the following query.
            
            Student Query: {query}
            Category: {category}
            
            Available Context from Knowledge Base:
            {context_text}
            
            Instructions:
            1. Provide a helpful, accurate response based on the context provided
            2. If the context doesn't fully address the query, acknowledge this
            3. Be specific and actionable in your response
            4. Use a friendly, professional tone
            5. If you're not confident about the answer, say so
            6. For technical issues, provide step-by-step guidance when possible
            
            Respond in JSON format:
            {{
                "response": "your detailed response here",
                "confidence_score": 0.XX (between 0.0 and 1.0 - how confident you are that this response fully addresses the query),
                "reasoning": "brief explanation of your confidence level"
            }}
            """
            
            response = await self.llm.ainvoke(response_prompt)
            
            try:
                result = json.loads(response.content)
                return result.get("response", "I'm having trouble processing your request."), result.get("confidence_score", 0.1)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in response generation: {str(e)}")
                logger.debug(f"Raw response: {response.content}")
                return (
                    "I'm having trouble generating a response right now. Your query has been forwarded to our support team.", 
                    0.1
                )
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return (
                "I'm having trouble generating a response right now. Your query has been forwarded to our support team who will get back to you soon.", 
                0.1
            )