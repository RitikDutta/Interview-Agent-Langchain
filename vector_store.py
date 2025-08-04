# Vector Store Module for AI Mock Interview Application
import os
import uuid
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime

# Pinecone and embedding imports
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# PDF processing imports
from langchain_community.document_loaders import PyPDFLoader
import glob

class VectorStore:
    """
    Vector database manager for AI Mock Interview Application using Pinecone.
    Handles resume storage, question bank, user responses, and semantic search.
    """
    
    def __init__(self, api_key: str, index_name: str = "interview-ai-db"):
        """
        Initialize the vector store with Pinecone configuration.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index to use
        """
        self.api_key = api_key
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)
        
        # Initialize embedding model - using free sentence-transformers model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Pinecone index
        self._setup_index()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_index(self):
        """Setup Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            if not self.pc.has_index(self.index_name):
                self.logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index with appropriate dimensions for sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                import time
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                    
            # Get index reference
            self.index = self.pc.Index(self.index_name)
            
            # Initialize LangChain vector store
            self.vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                text_key="text",
                namespace="default"
            )
            
            self.logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            self.logger.error(f"Error setting up Pinecone index: {str(e)}")
            raise
    
    def store_resume_chunks(self, user_id: str, resume_text: str, user_domain: str = None) -> List[str]:
        """
        Store resume content as chunks in vector database.
        
        Args:
            user_id: Unique user identifier
            resume_text: Extracted resume text
            user_domain: User's professional domain
            
        Returns:
            List of stored chunk IDs
        """
        try:
            # Split resume into chunks
            chunks = self.text_splitter.split_text(resume_text)
            
            documents = []
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"resume_{user_id}_{i}_{uuid.uuid4().hex[:8]}"
                chunk_ids.append(chunk_id)
                
                # Create document with metadata
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "user_id": user_id,
                        "content_type": "resume",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "domain": user_domain.lower() if user_domain else "general",
                        "timestamp": datetime.now().isoformat(),
                        "source": "resume_upload"
                    }
                )
                documents.append(doc)
            
            # Store in vector database using namespace for user isolation
            self.vector_store.add_documents(
                documents=documents,
                ids=chunk_ids,
                namespace=f"user_{user_id}"
            )
            
            self.logger.info(f"Stored {len(chunks)} resume chunks for user {user_id}")
            return chunk_ids
            
        except Exception as e:
            self.logger.error(f"Error storing resume chunks: {str(e)}")
            raise
    
    def search_questions(self, query: str, top_k: int = 5, filter_metadata: Dict = None) -> List[Dict[str, Any]]:
        """
        Search for relevant interview questions based on query and filters.
        
        Args:
            query: Search query (enhanced with user context)
            top_k: Number of results to return
            filter_metadata: Metadata filters for search
            
        Returns:
            List of relevant question results
        """
        try:
            # Clean filter metadata (remove None values)
            if filter_metadata:
                filter_metadata = {k: v for k, v in filter_metadata.items() if v is not None}
                
            # Search in questions namespace
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                namespace="questions",
                filter=filter_metadata if filter_metadata else None
            )
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            self.logger.info(f"Found {len(formatted_results)} questions for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching questions: {str(e)}")
            return []
    
    def store_user_response(self, user_id: str, text: str, metadata: Dict[str, Any]) -> str:
        """
        Store user's interview response for future analysis.
        
        Args:
            user_id: User identifier
            text: User's response text
            metadata: Additional metadata about the response
            
        Returns:
            ID of stored response
        """
        try:
            response_id = f"response_{user_id}_{uuid.uuid4().hex[:8]}"
            
            # Add system metadata
            metadata.update({
                "user_id": user_id,
                "content_type": "user_response",
                "timestamp": datetime.now().isoformat(),
                "response_id": response_id
            })
            
            # Create document
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            
            # Store in user's namespace
            self.vector_store.add_documents(
                documents=[doc],
                ids=[response_id],
                namespace=f"user_{user_id}_responses"
            )
            
            self.logger.info(f"Stored response {response_id} for user {user_id}")
            return response_id
            
        except Exception as e:
            self.logger.error(f"Error storing user response: {str(e)}")
            raise
    
    def get_user_response_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about user's response patterns.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with response statistics
        """
        try:
            # Query user's responses namespace
            namespace = f"user_{user_id}_responses"
            
            # Get sample of responses for analysis
            sample_query = "interview response analysis feedback"
            results = self.vector_store.similarity_search(
                query=sample_query,
                k=50,  # Analyze up to 50 recent responses
                namespace=namespace
            )
            
            if not results:
                return {"total_responses": 0, "analysis": "No responses found"}
            
            # Analyze response patterns
            domains = {}
            difficulties = {}
            timestamps = []
            
            for doc in results:
                meta = doc.metadata
                domain = meta.get("domain", "unknown")
                difficulty = meta.get("difficulty", "unknown")
                timestamp = meta.get("timestamp")
                
                domains[domain] = domains.get(domain, 0) + 1
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                
                if timestamp:
                    timestamps.append(timestamp)
            
            return {
                "total_responses": len(results),
                "domain_distribution": domains,
                "difficulty_distribution": difficulties,
                "recent_activity": len(timestamps),
                "analysis": f"User has responded to {len(results)} questions across {len(domains)} domains"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user response stats: {str(e)}")
            return {"error": str(e)}
    
    def search_user_context(self, user_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search user's personal context (resume, past responses) for relevant information.
        
        Args:
            user_id: User identifier
            query: Context search query
            top_k: Number of results to return
            
        Returns:
            List of relevant context from user's data
        """
        try:
            results = []
            
            # Search user's resume chunks
            try:
                resume_results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k,
                    namespace=f"user_{user_id}"
                )
                
                for doc, score in resume_results:
                    results.append({
                        "text": doc.page_content,
                        "source": "resume",
                        "metadata": doc.metadata,
                        "similarity_score": score
                    })
            except:
                pass  # No resume data found
            
            # Search user's response history
            try:
                response_results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k,
                    namespace=f"user_{user_id}_responses"
                )
                
                for doc, score in response_results:
                    results.append({
                        "text": doc.page_content,
                        "source": "past_response",
                        "metadata": doc.metadata,
                        "similarity_score": score
                    })
            except:
                pass  # No response history found
            
            # Sort by similarity score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error searching user context: {str(e)}")
            return []
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data associated with a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Success status
        """
        try:
            namespaces_to_clear = [
                f"user_{user_id}",
                f"user_{user_id}_responses"
            ]
            
            for namespace in namespaces_to_clear:
                try:
                    # Delete all vectors in namespace
                    self.index.delete(delete_all=True, namespace=namespace)
                    self.logger.info(f"Cleared namespace: {namespace}")
                except:
                    pass  # Namespace might not exist
            
            self.logger.info(f"Deleted all data for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting user data: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database index"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            self.logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}


class QuestionBankLoader:
    """
    Separate module for loading interview questions from PDF files into vector database.
    Run this separately to populate the question bank.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize question bank loader.
        
        Args:
            vector_store: VectorStore instance to use for storage
        """
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
    
    def load_questions_from_pdf_directory(self, pdf_directory: str) -> Dict[str, Any]:
        """
        Load interview questions from all PDF files in a directory.
        
        Args:
            pdf_directory: Path to directory containing PDF files
            
        Returns:
            Loading results summary
        """
        try:
            if not os.path.exists(pdf_directory):
                raise FileNotFoundError(f"Directory not found: {pdf_directory}")
            
            # Find all PDF files
            pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
            
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in {pdf_directory}")
            
            self.logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            total_questions = 0
            processed_files = []
            errors = []
            
            for pdf_file in pdf_files:
                try:
                    result = self.load_questions_from_pdf(pdf_file)
                    total_questions += result["questions_stored"]
                    processed_files.append({
                        "file": os.path.basename(pdf_file),
                        "questions": result["questions_stored"]
                    })
                    self.logger.info(f"Processed {pdf_file}: {result['questions_stored']} questions")
                    
                except Exception as e:
                    error_msg = f"Error processing {pdf_file}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            return {
                "total_files_processed": len(processed_files),
                "total_questions_stored": total_questions,
                "processed_files": processed_files,
                "errors": errors,
                "success": len(errors) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Error loading questions from directory: {str(e)}")
            return {"error": str(e), "success": False}
    
    def load_questions_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Load interview questions from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Loading results for the file
        """
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Combine all pages
            full_text = "\n\n".join([page.page_content for page in pages])
            
            # Extract metadata from filename
            filename = os.path.basename(pdf_path)
            domain, difficulty, category = self._extract_metadata_from_filename(filename)
            
            # Split into question chunks
            chunks = self.vector_store.text_splitter.split_text(full_text)
            
            documents = []
            question_ids = []
            
            for i, chunk in enumerate(chunks):
                # Skip chunks that are too short or don't look like questions
                if len(chunk.strip()) < 50:
                    continue
                
                question_id = f"question_{uuid.uuid4().hex[:12]}"
                question_ids.append(question_id)
                
                # Create document with rich metadata
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        "content_type": "interview_question",
                        "source_file": filename,
                        "source_path": pdf_path,
                        "domain": domain,
                        "difficulty": difficulty,
                        "category": category,
                        "chunk_index": i,
                        "question_id": question_id,
                        "timestamp": datetime.now().isoformat(),
                        "length": len(chunk.strip())
                    }
                )
                documents.append(doc)
            
            # Store in questions namespace
            if documents:
                self.vector_store.vector_store.add_documents(
                    documents=documents,
                    ids=question_ids,
                    namespace="questions"
                )
            
            return {
                "questions_stored": len(documents),
                "source_file": filename,
                "domain": domain,
                "difficulty": difficulty,
                "category": category
            }
            
        except Exception as e:
            self.logger.error(f"Error loading questions from PDF: {str(e)}")
            raise
    
    def _extract_metadata_from_filename(self, filename: str) -> tuple:
        """
        Extract domain, difficulty, and category from filename.
        
        Expected format: domain_difficulty_category.pdf
        Example: python_intermediate_technical.pdf
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Tuple of (domain, difficulty, category)
        """
        try:
            # Remove .pdf extension
            base_name = filename.replace('.pdf', '').lower()
            
            # Split by underscores
            parts = base_name.split('_')
            
            # Define possible values
            domains = ['python', 'java', 'javascript', 'data-science', 'machine-learning', 
                      'software-engineering', 'web-development', 'system-design', 'general']
            difficulties = ['beginner', 'intermediate', 'advanced', 'expert']
            categories = ['technical', 'behavioral', 'situational', 'coding', 'system-design']
            
            domain = 'general'
            difficulty = 'intermediate'
            category = 'technical'
            
            # Extract domain
            for part in parts:
                if part in domains:
                    domain = part
                    break
            
            # Extract difficulty
            for part in parts:
                if part in difficulties:
                    difficulty = part
                    break
            
            # Extract category
            for part in parts:
                if part in categories:
                    category = part
                    break
            
            return domain, difficulty, category
            
        except:
            # Default values if parsing fails
            return 'general', 'intermediate', 'technical'
    
    def test_question_search(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """
        Test question search functionality with sample queries.
        
        Args:
            test_queries: List of test queries (optional)
            
        Returns:
            Test results
        """
        if test_queries is None:
            test_queries = [
                "Python programming loops and functions",
                "JavaScript async await promises",
                "Machine learning algorithms explanation",
                "System design scalability questions",
                "Behavioral interview leadership experience"
            ]
        
        results = {}
        
        for query in test_queries:
            try:
                search_results = self.vector_store.search_questions(query, top_k=3)
                results[query] = {
                    "found_questions": len(search_results),
                    "sample_question": search_results[0]["text"][:200] + "..." if search_results else "No results",
                    "success": True
                }
            except Exception as e:
                results[query] = {
                    "error": str(e),
                    "success": False
                }
        
        return results


# Example usage and initialization script
def initialize_vector_database(api_key: str) -> VectorStore:
    """
    Initialize vector database for the interview application.
    
    Args:
        api_key: Pinecone API key
        
    Returns:
        Configured VectorStore instance
    """
    try:
        vector_store = VectorStore(api_key=api_key)
        print("✅ Vector database initialized successfully")
        
        # Display index stats
        stats = vector_store.get_index_stats()
        print(f"📊 Index stats: {stats}")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error initializing vector database: {str(e)}")
        raise


def load_question_bank(vector_store: VectorStore, pdf_directory: str) -> None:
    """
    Load question bank from PDF directory.
    
    Args:
        vector_store: VectorStore instance
        pdf_directory: Path to directory containing question PDFs
    """
    try:
        loader = QuestionBankLoader(vector_store)
        results = loader.load_questions_from_pdf_directory(pdf_directory)
        
        if results["success"]:
            print(f"✅ Successfully loaded {results['total_questions_stored']} questions from {results['total_files_processed']} files")
            for file_info in results["processed_files"]:
                print(f"   - {file_info['file']}: {file_info['questions']} questions")
        else:
            print(f"❌ Error loading questions: {results.get('error', 'Unknown error')}")
            if results.get("errors"):
                for error in results["errors"]:
                    print(f"   - {error}")
        
        # Test search functionality
        print("\n🔍 Testing question search...")
        test_results = loader.test_question_search()
        for query, result in test_results.items():
            if result["success"]:
                print(f"   ✅ '{query}': Found {result['found_questions']} questions")
            else:
                print(f"   ❌ '{query}': {result['error']}")
                
    except Exception as e:
        print(f"❌ Error loading question bank: {str(e)}")
        raise


# Main execution script
if __name__ == "__main__":
    """
    Run this script to initialize the vector database and load question bank.
    
    Usage:
    1. Set PINECONE_API_KEY environment variable
    2. Place PDF files with interview questions in a directory
    3. Run: python vector_store.py
    """
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        exit(1)
    
    # Initialize vector database
    print("🚀 Initializing vector database...")
    vector_store = initialize_vector_database(api_key)
    
    # Load question bank (customize this path)
    pdf_directory = "./interview_questions_pdfs"  # Change to your PDF directory path
    
    if os.path.exists(pdf_directory):
        print(f"\n📚 Loading question bank from {pdf_directory}...")
        load_question_bank(vector_store, pdf_directory)
    else:
        print(f"⚠️  PDF directory not found: {pdf_directory}")
        print("   Create this directory and add PDF files with interview questions")
        print("   Expected filename format: domain_difficulty_category.pdf")
        print("   Example: python_intermediate_technical.pdf")
    
    print("\n🎉 Setup complete! Your vector database is ready for the interview application.")