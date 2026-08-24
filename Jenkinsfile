pipeline {
    agent any

    triggers {
    pollSCM('H/5 * * * *')
    }

    options {
        ansiColor('xterm')
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    environment {
        AWS_ACCESS_KEY_ID = credentials('aws-access-key-id')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-access-key')
        AWS_DEFAULT_REGION = 'ap-south-1'
        TF_IN_AUTOMATION = 'true'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Validate') {
            steps {
                bat 'terraform fmt -check -recursive -diff'
                bat 'terraform init -input=false'
                bat 'terraform validate'
            }
        }

        stage('Security Scan') {
            steps {
                bat 'tflint --init'
                bat 'tflint --format compact'
                bat 'tfsec . --format junit --out tfsec-report.xml --soft-fail'
                bat 'tfsec . --minimum-severity HIGH'
            }

            post {
                always {
                    junit allowEmptyResults: true, testResults: 'tfsec-report.xml'
                }
            }
        }

        stage('Plan') {
            steps {
                bat 'terraform plan -input=false -out=tfplan'
                bat 'terraform show -no-color tfplan > tfplan.txt'

                archiveArtifacts artifacts: 'tfplan, tfplan.txt',
                                 fingerprint: true
            }
        }

        stage('Approval') {
            when {
                branch 'main'
            }

            steps {
                timeout(time: 30, unit: 'MINUTES') {
                    input(
                        message: 'Apply the archived plan to the cloud account?',
                        ok: 'Apply'
                    )
                }
            }
        }

        stage('Apply') {
            when {
                branch 'main'
            }

            steps {
                bat 'terraform apply -input=false tfplan'
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully.'
        }

        failure {
            echo 'Pipeline failed - inspect the stage that went red.'
        }

        always {
            deleteDir()
        }
    }
}
