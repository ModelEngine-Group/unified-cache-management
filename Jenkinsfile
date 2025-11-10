pipeline {
    agent any

    environment {
        DOCKER_REGISTRY  = '127.0.0.1:5001'
        REPO_URL         = 'https://github.com/ModelEngine-Group/unified-cache-management.git'
        REPORT_DIR       = 'tests/reports'
        EMAIL_RECIPIENTS = 'duxiaolong22@mails.ucas.ac.cn'
        DEVICE           = '/dev/davinci7'
        MODEL_PATH       = '/home/models/Qwen3-0.6B'
		PALTFORM         = 'vllm-ascend'
    }

    parameters {
        string(name: 'MANUAL_BRANCH', defaultValue: 'main', description: 'Branch to build when triggered manually')
        choice(name: 'TRIGGER_TYPE', choices: ['PR_VERIFY', 'DAILY_BUILD', 'FULL_TEST'], description: 'Manual build type')
        booleanParam(name: 'SEND_EMAIL', defaultValue: true, description: 'Send e-mail notification?')
    }

    triggers {
        // Daily build at 00:00
        cron('H 0 * * *')
    }

    stages {
        /* ---------------------------------------------------------- */
        /* 1. PR Verification                                         */
        /* ---------------------------------------------------------- */
        stage('PR Verification') {
            when {
                // Run ONLY when the branch name starts with “PR-”
                // OR when a manual build explicitly chooses PR_VERIFY.
                // exclude timer-triggered builds
                anyOf {
                    expression { env.CHANGE_ID != null }          // multibranch PR
                    allOf {
                        expression { params.TRIGGER_TYPE == 'PR_VERIFY' }
                        not { triggeredBy 'TimerTrigger' }
                    }
                }
            }
            steps {
                script {
                    echo '===== Starting PR Verification ====='
                    checkout scm
                    runSimpleTest()
                }
            }
            post {
                always {
                    generateSimpleReports()
                    cleanupResources()
                }
                success {
                    sendNotification('PR Verification', 'SUCCESS')
                }
                failure {
                    sendNotification('PR Verification', 'FAILURE')
                }
            }
        }

        /* ---------------------------------------------------------- */
        /* 2. Daily Build & Smoke Test                               */
        /* ---------------------------------------------------------- */
        stage('Daily Build & Smoke Test') {
            when {
                anyOf {
                    triggeredBy 'TimerTrigger'
                    expression { params.TRIGGER_TYPE == 'DAILY_BUILD' }
                }
            }
            steps {
                script {
                    echo '===== Starting Daily Build & Smoke Test ====='
                    checkout([$class: 'GitSCM',
                              branches: [[name: '*/develop']],
                              userRemoteConfigs: [[url: "${REPO_URL}"]]])
                    buildSimpleImage('hello-world-service')
                    runSimpleTest()
                }
            }
            post {
                always {
                    generateSimpleReports()
                    cleanupResources()
                }
                success {
                    sendNotification('Daily Build & Smoke Test', 'SUCCESS')
                }
                failure {
                    sendNotification('Daily Build & Smoke Test', 'FAILURE')
                }
            }
        }

        /* ---------------------------------------------------------- */
        /* 3. Full Test                                               */
        /* ---------------------------------------------------------- */
        stage('Full Test') {
            when { expression { params.TRIGGER_TYPE == 'FULL_TEST' } }
            steps {
                script {
                    echo '===== Starting Full Test ====='
                    checkout([$class: 'GitSCM',
                              branches: [[name: '*/develop']],
                              userRemoteConfigs: [[url: "${REPO_URL}"]]])
                    buildSimpleImage('hello-world-service')
                    runSimpleTest()
                }
            }
            post {
                always {
                    generateSimpleReports()
                    cleanupResources()
                }
                success {
                    sendNotification('Full Test', 'SUCCESS')
                }
                failure {
                    sendNotification('Full Test', 'FAILURE')
                }
            }
        }

        /* ---------------------------------------------------------- */
        /* 4. Collect Reports                                         */
        /* ---------------------------------------------------------- */
        stage('Collect Reports') {
            steps {
                script {
                    echo 'Collecting all logs and reports...'
                    sh "mkdir -p '${REPORT_DIR}'"

                    archiveArtifacts artifacts: 'tests/reports/**', allowEmptyArchive: true

                    publishHTML([allowMissing: false,
                                 alwaysLinkToLastBuild: true,
                                 keepAll: true,
                                 reportDir: 'tests/reports',
                                 reportFiles: 'test_report.html',
                                 reportName: 'Test Report',
                                 reportTitles: 'Pytest Test Results'])
                }
            }
        }
    }

    post {
        always {
            echo "Pipeline Finished: ${currentBuild.currentResult}"
            cleanWs()
        }
    }
}

// ======================================================================
// Helper functions
// ======================================================================

/* Build a docker image and tag it */
def buildSimpleImage(String imageName) {
    def fullImageName = "${DOCKER_REGISTRY}/${imageName}"
    def dateTag  = "daily-${new Date().format('yyyyMMdd')}"
    def gitCommit = sh(returnStdout: true, script: 'git rev-parse --short HEAD').trim()

    echo "Building Docker image: ${fullImageName}"

    sh """
    docker build -t ${fullImageName}:${dateTag} .
    docker tag ${fullImageName}:${dateTag} ${fullImageName}:${gitCommit}
    docker tag ${fullImageName}:${dateTag} ${fullImageName}:latest
    """
}

/* Run the test suite inside docker-compose */
def runSimpleTest() {
    echo '===== Running Simple Test Suite ====='
    try {
        sh 'ls -la'
        sh 'pwd'

        sh '''
        cd /test
        pytest --stage=0 > build.log
        '''

        sh '''
        echo "=== Check Report results ==="
        ls -la tests/reports/              || echo "tests/reports missing"
        echo "=== Check complete ==="
        '''
    } catch (Exception e) {
        echo "Test execution failed: ${e.message}"
        sh 'cat build.log || true'
        currentBuild.result = 'FAILURE'
        throw e
    }
}

/* Collect test artefacts into REPORT_DIR */
def generateSimpleReports() {
    echo 'Generating test reports...'
    sh "mkdir -p '${REPORT_DIR}'"

    sh '''
    echo "=== Collect test reports ==="
    ls -la tests/reports/ 2>/dev/null || echo "No reports directory"
    cp -r tests/reports/* "${REPORT_DIR}/" 2>/dev/null || echo "No reports to copy"

    '''
}

/* Send e-mail notification */
def sendNotification(String testType, String status) {
    if (!params.SEND_EMAIL) return

    echo "Sending e-mail notification for ${testType} - ${status}"

    def subject = "[Jenkins] ${testType} - ${status} - ${env.JOB_NAME} #${env.BUILD_NUMBER}"
	def body = """
	<html><body>
	  <h2>Jenkins Build Notification</h2>

	  <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
		<tr><td><strong>Project:</strong></td><td>${env.JOB_NAME}</td></tr>
		<tr><td><strong>Build Number:</strong></td><td>#${env.BUILD_NUMBER}</td></tr>
		<tr><td><strong>Status:</strong></td><td style="color:${status == 'SUCCESS' ? 'green' : 'red'}; font-weight:bold;">${status}</td></tr>
		<tr><td><strong>Test Type:</strong></td><td>${testType}</td></tr>
		<tr><td><strong>Build Time:</strong></td><td>${new Date().format("yyyy-MM-dd HH:mm:ss")}</td></tr>
		<tr><td><strong>Commit ID:</strong></td><td>${sh(returnStdout: true, script: 'git rev-parse HEAD').trim()}</td></tr>
		<tr><td><strong>Commit Message:</strong></td><td>${sh(returnStdout: true, script: 'git log --oneline -1').trim()}</td></tr>
		<tr><td><strong>Build URL:</strong></td><td><a href="${env.BUILD_URL}">${env.BUILD_URL}</a></td></tr>
	  </table>

	  <h3>Test Summary</h3>
	  <p>Please refer to the attached artifacts or click on the links above for detailed results.</p>

	  <p>Best regards,<br/>Jenkins CI/CD System</p>
	</body></html>
	"""

    try {
        emailext subject: subject,
                 body: body,
                 to: "${EMAIL_RECIPIENTS}",
                 mimeType: 'text/html'
    } catch (Exception e) {
        echo "Failed to send e-mail: ${e.message}"
    }
}

/* Clean up containers and dangling images */
def cleanupResources() {
    echo 'Cleaning up resources...'
    sh '''
    docker-compose -f docker-compose.test.yml down -v || true
    docker system prune -f || true
    '''
}