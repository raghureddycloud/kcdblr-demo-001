kubectl get svc argocd-server -n argocd
echo "=========================================================================================="
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo
echo "=========================================================================================="