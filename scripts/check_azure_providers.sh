#!/bin/bash

# Azure Resource Provider Registration Checker
# This script checks the registration status of all providers required for Azure ML deployment

echo "🔍 Azure Resource Provider Registration Status"
echo "=============================================="
echo ""

# Define required providers for Azure ML
CORE_PROVIDERS=(
    "Microsoft.MachineLearningServices"
    "Microsoft.ContainerRegistry"
    "Microsoft.Storage"
    "Microsoft.KeyVault"
    "Microsoft.Insights"
)

ADDITIONAL_PROVIDERS=(
    "Microsoft.ContainerInstance"
    "Microsoft.Web"
    "Microsoft.Network"
    "Microsoft.Compute"
    "Microsoft.Cdn"
    "Microsoft.ServiceBus"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if Azure CLI is installed and authenticated
check_azure_cli() {
    if ! command -v az &> /dev/null; then
        echo -e "${RED}❌ Azure CLI is not installed${NC}"
        echo "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    
    # Check if logged in
    if ! az account show &> /dev/null; then
        echo -e "${RED}❌ Not logged in to Azure${NC}"
        echo "Run: az login"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Azure CLI is installed and authenticated${NC}"
    echo ""
}

# Function to check provider status
check_provider() {
    local provider=$1
    local status=$(az provider show --namespace "$provider" --query "registrationState" -o tsv 2>/dev/null)
    
    if [ -z "$status" ]; then
        echo -e "${RED}❌ $provider - Unknown/Error${NC}"
        return 1
    elif [ "$status" = "Registered" ]; then
        echo -e "${GREEN}✅ $provider - $status${NC}"
        return 0
    elif [ "$status" = "Registering" ]; then
        echo -e "${YELLOW}⏳ $provider - $status${NC}"
        return 2
    else
        echo -e "${RED}❌ $provider - $status${NC}"
        return 1
    fi
}

# Function to register a provider
register_provider() {
    local provider=$1
    echo "Registering $provider..."
    az provider register --namespace "$provider"
}

# Main execution
main() {
    check_azure_cli
    
    echo "Current Azure Subscription:"
    az account show --query "{Name:name, Id:id, TenantId:tenantId}" -o table
    echo ""
    
    echo -e "${BLUE}📋 Core Azure ML Providers:${NC}"
    echo "=========================="
    
    local core_failed=0
    for provider in "${CORE_PROVIDERS[@]}"; do
        check_provider "$provider"
        if [ $? -eq 1 ]; then
            ((core_failed++))
        fi
    done
    echo ""
    
    echo -e "${BLUE}📋 Additional Providers for Advanced Features:${NC}"
    echo "============================================="
    
    local additional_failed=0
    for provider in "${ADDITIONAL_PROVIDERS[@]}"; do
        check_provider "$provider"
        if [ $? -eq 1 ]; then
            ((additional_failed++))
        fi
    done
    echo ""
    
    # Summary
    echo -e "${BLUE}📊 Summary:${NC}"
    echo "==========="
    
    if [ $core_failed -eq 0 ]; then
        echo -e "${GREEN}✅ All core providers are registered${NC}"
    else
        echo -e "${RED}❌ $core_failed core provider(s) need registration${NC}"
    fi
    
    if [ $additional_failed -eq 0 ]; then
        echo -e "${GREEN}✅ All additional providers are registered${NC}"
    else
        echo -e "${YELLOW}⚠️  $additional_failed additional provider(s) need registration${NC}"
    fi
    echo ""
    
    # Provide registration commands if needed
    if [ $core_failed -gt 0 ] || [ $additional_failed -gt 0 ]; then
        echo -e "${BLUE}🔧 Registration Commands:${NC}"
        echo "========================"
        echo ""
        
        if [ $core_failed -gt 0 ]; then
            echo "# Register missing core providers:"
            for provider in "${CORE_PROVIDERS[@]}"; do
                local status=$(az provider show --namespace "$provider" --query "registrationState" -o tsv 2>/dev/null)
                if [ "$status" != "Registered" ]; then
                    echo "az provider register --namespace $provider"
                fi
            done
            echo ""
        fi
        
        if [ $additional_failed -gt 0 ]; then
            echo "# Register missing additional providers:"
            for provider in "${ADDITIONAL_PROVIDERS[@]}"; do
                local status=$(az provider show --namespace "$provider" --query "registrationState" -o tsv 2>/dev/null)
                if [ "$status" != "Registered" ]; then
                    echo "az provider register --namespace $provider"
                fi
            done
            echo ""
        fi
        
        echo "# Register all at once:"
        echo "az provider register --namespace Microsoft.MachineLearningServices \\"
        echo "  --namespace Microsoft.ContainerRegistry \\"
        echo "  --namespace Microsoft.Storage \\"
        echo "  --namespace Microsoft.KeyVault \\"
        echo "  --namespace Microsoft.Insights \\"
        echo "  --namespace Microsoft.ContainerInstance \\"
        echo "  --namespace Microsoft.Web \\"
        echo "  --namespace Microsoft.Network \\"
        echo "  --namespace Microsoft.Compute \\"
        echo "  --namespace Microsoft.Cdn \\"
        echo "  --namespace Microsoft.ServiceBus"
        echo ""
    fi
    
    # Show all unregistered providers in subscription
    echo -e "${BLUE}🔍 All Unregistered Providers in Subscription:${NC}"
    echo "============================================="
    az provider list --query "[?registrationState=='NotRegistered'].{Namespace:namespace, State:registrationState}" -o table
    echo ""
    
    # Show registering providers (in progress)
    echo -e "${BLUE}⏳ Currently Registering Providers:${NC}"
    echo "=================================="
    local registering=$(az provider list --query "[?registrationState=='Registering'].{Namespace:namespace, State:registrationState}" -o table)
    if [ -z "$registering" ] || [ "$registering" = "Namespace    State" ]; then
        echo "None"
    else
        echo "$registering"
    fi
    echo ""
    
    echo -e "${BLUE}💡 Tips:${NC}"
    echo "======="
    echo "• Provider registration can take 5-10 minutes"
    echo "• Re-run this script to check registration progress"
    echo "• Use 'az provider register --namespace <provider>' to register individually"
    echo "• Some providers may not be available in all regions"
}

# Handle command line options
case "${1:-}" in
    --help|-h)
        echo "Azure Resource Provider Registration Checker"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --register     Automatically register missing core providers"
        echo "  --all          Show all providers in subscription"
        echo ""
        echo "This script checks the registration status of Azure resource providers"
        echo "required for Azure Machine Learning deployment."
        exit 0
        ;;
    --register)
        echo "🔧 Auto-registering missing core providers..."
        check_azure_cli
        for provider in "${CORE_PROVIDERS[@]}"; do
            local status=$(az provider show --namespace "$provider" --query "registrationState" -o tsv 2>/dev/null)
            if [ "$status" != "Registered" ]; then
                register_provider "$provider"
            fi
        done
        echo ""
        echo "✅ Registration commands sent. Re-run without --register to check status."
        exit 0
        ;;
    --all)
        echo "📋 All Azure Resource Providers:"
        echo "==============================="
        az provider list --query "sort_by([].{Namespace:namespace, State:registrationState}, &Namespace)" -o table
        exit 0
        ;;
    *)
        main
        ;;
esac