//
// Created by Martin Steinegger on 11/14/20.
//
#include "Command.h"
#include "LocalParameters.h"
#include "Debug.h"
#include "DBWriter.h"

#include "structureto3di.h"
#include "SubstitutionMatrix.h"
#include "GemmiWrapper.h"

#include <iostream>
#include <dirent.h>

#ifdef OPENMP
#include <omp.h>
#endif


int createdb(int argc, const char **argv, const Command& command) {
    LocalParameters& par = LocalParameters::getLocalInstance();
    par.parseParameters(argc, argv, command, true, 0, MMseqsParameter::COMMAND_COMMON);

    std::vector<std::string> filenames(par.filenames);
    std::string outputName = filenames.back();
    filenames.pop_back();
    if(filenames.size() == 1 && FileUtil::directoryExists(filenames.back().c_str())){
        std::vector<std::string> dirs;
        dirs.push_back(filenames.back());
        filenames.pop_back();
        while(dirs.size() != 0){
            std::string dir = dirs.back();
            dirs.pop_back();
            DIR * dpdf = opendir(dir.c_str());
            if (dpdf != NULL) {
                while (dirent * epdf = readdir(dpdf)) {
                    std::string filename(epdf->d_name);
                    if(filename != "." && filename !=".."){
                        if (epdf->d_type == DT_DIR){
                            dirs.push_back(dir+"/"+filename);
                        }else{
                            filenames.push_back(dir+"/"+filename);
                        }
                    }
                }
                closedir(dpdf);
            }
        }
    }
    Debug(Debug::INFO) << "Output file: " << par.db2 << "\n";

    DBWriter torsiondbw((outputName+"_ss").c_str(), (outputName+"_ss.index").c_str(), static_cast<unsigned int>(par.threads), par.compressed, Parameters::DBTYPE_AMINO_ACIDS);
    torsiondbw.open();
    DBWriter hdbw((outputName+"_h").c_str(), (outputName+"_h.index").c_str(), static_cast<unsigned int>(par.threads), par.compressed, Parameters::DBTYPE_GENERIC_DB);
    hdbw.open();
    DBWriter cadbw((outputName+"_ca").c_str(), (outputName+"_ca.index").c_str(), static_cast<unsigned int>(par.threads), par.compressed, LocalParameters::DBTYPE_CA_ALPHA);
    cadbw.open();
    DBWriter aadbw((outputName).c_str(), (outputName+".index").c_str(), static_cast<unsigned int>(par.threads), par.compressed, Parameters::DBTYPE_AMINO_ACIDS);
    aadbw.open();
    SubstitutionMatrix mat(par.scoringMatrixFile.aminoacids, 2.0, par.scoreBias);
    Debug::Progress progress(filenames.size());

    size_t incorrectFiles = 0;
    //===================== single_process ===================//__110710__//
#pragma omp parallel default(none) shared(par, torsiondbw, hdbw, cadbw, aadbw, mat, filenames, progress) reduction(+:incorrectFiles)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        //recon_related
        StructureTo3Di structureTo3Di;
        GemmiWrapper readStructure;
        std::vector<char> alphabet3di;
        std::vector<float> camol;

        std::string name;
#pragma omp for schedule(static)
        for (size_t i = 0; i < filenames.size(); i++) {
            progress.updateProgress();
            // clear memory
            alphabet3di.clear();
            camol.clear();
            if(readStructure.load(filenames[i]) == false){
                incorrectFiles++;
                continue;
            }
            for(size_t ch = 0; ch < readStructure.chain.size(); ch++){
                size_t chainStart = readStructure.chain[ch].first;
                size_t chainEnd = readStructure.chain[ch].second;
                size_t chainLen = chainEnd - chainStart;
                char * states = structureTo3Di.structure2states(&readStructure.ca[chainStart],
                                                                &readStructure.n[chainStart],
                                                                &readStructure.c[chainStart],
                                                                &readStructure.cb[chainStart],
                                                                chainLen);
                for(size_t pos = 0; pos < chainLen; pos++){
                    alphabet3di.push_back(mat.num2aa[static_cast<int>(states[pos])]);
                }
                alphabet3di.push_back('\n');
                torsiondbw.writeData(alphabet3di.data(), alphabet3di.size(), i, thread_idx);
                aadbw.writeStart(thread_idx);
                aadbw.writeAdd(&readStructure.ami[chainStart], chainLen, thread_idx);
                char newline = '\n';
                aadbw.writeAdd(&newline, 1, thread_idx);
                aadbw.writeEnd(i, thread_idx);
                hdbw.writeData(readStructure.names[ch].c_str(), readStructure.names[ch].size(), i, thread_idx);
                name.clear();
                for(size_t pos = 0; pos < chainLen; pos++){
                    camol.push_back(readStructure.ca[chainStart+pos].x);
                }
                for(size_t pos = 0; pos < chainLen; pos++) {
                    camol.push_back(readStructure.ca[chainStart+pos].y);
                }
                for(size_t pos = 0; pos < chainLen; pos++) {
                    camol.push_back(readStructure.ca[chainStart+pos].z);
                }
                cadbw.writeData((const char*)camol.data(), camol.size() * sizeof(float), i, thread_idx);
            }


        }
    }

    torsiondbw.close(true);
    hdbw.close(true);
    cadbw.close(true);
    aadbw.close(true);

    if (par.writeLookup == true) {
        DBReader<unsigned int> readerHeader((outputName+"_h").c_str(), (outputName+"_h.index").c_str(),
                                            1, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
        readerHeader.open(DBReader<unsigned int>::NOSORT);
        // create lookup file
        std::string sourceFilename = outputName + ".source";
        FILE* sourceFile = FileUtil::openAndDelete(sourceFilename.c_str(), "w");
        size_t prevFileKey = SIZE_MAX;
        std::string lookupFile = outputName + ".lookup";
        FILE* file = FileUtil::openAndDelete(lookupFile.c_str(), "w");
        std::string buffer;
        buffer.reserve(2048);
        DBReader<unsigned int>::LookupEntry entry;
        size_t globalCounter = 0;
        for (unsigned int id = 0; id < readerHeader.getSize(); id++) {
            size_t fileKey = readerHeader.getDbKey(id);
            char *header = readerHeader.getData(id, 0);
            entry.id = globalCounter;
            globalCounter++;
            entry.entryName = Util::parseFastaHeader(header);
            if (entry.entryName.empty()) {
                Debug(Debug::WARNING) << "Cannot extract identifier from entry " << id << "\n";
            }
            entry.fileNumber = fileKey;
            readerHeader.lookupEntryToBuffer(buffer, entry);
            size_t written = fwrite(buffer.c_str(), sizeof(char), buffer.size(), file);
            if (written != buffer.size()) {
                Debug(Debug::ERROR) << "Cannot write to lookup file " << lookupFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            buffer.clear();

            if(prevFileKey != fileKey){

                char sourceBuffer[4096];

                size_t len = snprintf(sourceBuffer, sizeof(sourceBuffer), "%zu\t%s\n", fileKey, FileUtil::baseName(filenames[fileKey].c_str()).c_str());
                written = fwrite(sourceBuffer, sizeof(char), len, sourceFile);
                if (written != len) {
                    Debug(Debug::ERROR) << "Cannot write to lookup file " << lookupFile << "\n";
                    EXIT(EXIT_FAILURE);
                }
            }
            prevFileKey = fileKey;
        }
        if (fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close file " << lookupFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        if(fclose(sourceFile) != 0){
            Debug(Debug::ERROR) << "Cannot close file " << sourceFilename << "\n";
            EXIT(EXIT_FAILURE);
        }
        readerHeader.close();
    }


    DBWriter::createRenumberedDB((outputName+"_ss").c_str(), (outputName+"_ss.index").c_str(), "", "", DBReader<unsigned int>::LINEAR_ACCCESS);
    DBWriter::createRenumberedDB((outputName+"_h").c_str(), (outputName+"_h.index").c_str(), "", "", DBReader<unsigned int>::LINEAR_ACCCESS);
    DBWriter::createRenumberedDB((outputName+"_ca").c_str(), (outputName+"_ca.index").c_str(), "", "", DBReader<unsigned int>::LINEAR_ACCCESS);
    DBWriter::createRenumberedDB((outputName).c_str(), (outputName+".index").c_str(), "", "", DBReader<unsigned int>::LINEAR_ACCCESS);



    Debug(Debug::INFO) << incorrectFiles << " out of " << filenames.size() << " entries are incorrect.\n";
    return EXIT_SUCCESS;
}
