#######################
# TODO (begin) #
#######################
# Change 'youruniqname' to match your UM uniqname (no quote marks).
UNIQNAME    = morag

# This is the path in Great Lakes to where projects will be
# uploaded. By default goes to user's home directory.
# Change this to select your preferred path.
# REMOTE_PATH := /nfs/turbo/coe-goroda/JANUS/GabrielMora/HallThrusterPem
REMOTE_PATH := /home/morag/HallThrusterPem

#######################
# TODO (end) #
#######################

# REMOTE_PATH has default definition above
sync2arc:
ifeq ($(UNIQNAME), youruniqname)
	@echo Edit UNIQNAME variable in Makefile.
	@exit 1;
endif
	# Synchronize local files into target directory on CAEN
	rsync \
      -av \
      --exclude '*.o' \
      --exclude '.git*' \
      --exclude '.vs*' \
      --exclude '*.code-workspace' \
      --filter=":- .gitignore" \
      "."/ \
      "$(UNIQNAME)@greatlakes.arc-ts.umich.edu:$(REMOTE_PATH)/"
	echo "Files synced to ARC at $(REMOTE_PATH)/"
.PHONY: sync2arc
# --delete \
