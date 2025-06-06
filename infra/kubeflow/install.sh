#https://github.com/kubeflow/manifests?tab=readme-ov-file#installation

# brew install kustomize 

#while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 20; done

#kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
