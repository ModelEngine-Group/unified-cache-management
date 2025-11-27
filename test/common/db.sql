CREATE TABLE test_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    test_case VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    error TEXT,
    test_build_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);