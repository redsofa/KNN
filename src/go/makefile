GOPATH := $(shell pwd)


all: build

go-deps:
	@echo Using Go Path :
	@echo $(GOPATH)	
	@echo NO DEPENDENCIES TO FETCH

dist-dir:
	@echo Creating dist directory
	mkdir ./dist

go-doc:
	@echo Launching Go docs server
	godoc -http=:6060
	$ (cd src/github.com/redsofa/; godoc -http=:6060)

build: clean dist-dir 
	@echo Building knn
	@echo Using Go Path :
	@echo $(GOPATH)
	mkdir ./dist/knn
	go build -o ./dist/knn/knn src/github.com/redsofa/knn/main/main.go

clean:
	@echo Cleaning ...
	rm -fR ./dist